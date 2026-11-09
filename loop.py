"""The research loop. Each turn:
  1. copy lab/ into a fresh, empty workspace
  2. a fresh Claude Code session reads program.md + ledger.tsv and changes model.py. It works
     in the agent sandbox: a container that sees only that workspace and reaches only Claude's API
  3. only model.py comes back; the judge compares it with the champion on the validation months,
     with a bar that rises with every idea tested (see judge.py)
  4. keep it (new champion) or throw it away, and write the verdict into ledger.tsv

    uv run loop.py run 5     # five more turns (continues from the ledger)
    uv run loop.py holdout   # score the holdout once, at the end: Jul - Dec 2025
    uv run loop.py reset     # back to the baseline
    uv run loop.py build     # build the Docker images (run and holdout do this first anyway)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
LAB, JUDGE, HISTORY = ROOT / "lab", ROOT / "judge", ROOT / "history"

# TODO: Does the ledger show amount of improvement, or just kept/rejected?
LEDGER = LAB / "ledger.tsv"
COLUMNS = ["turn", "idea", "val_mae", "win_pct", "bar_pct", "decision"]
TRIED = ("KEPT", "thrown away")  # the decisions that count as a try, so they raise the bar
AGENT_FILES = ["program.md", "model.py", "check.py", "data.csv", "ledger.tsv"]
IMAGES = ["model-sandbox", "agent-sandbox", "agent-proxy"]  # folders in docker/, each built as the Docker image toy-<name>
NETWORK = "toy-agent-net"  # the agent sandbox's network: its only way out is the proxy
PROXY = "toy-agent-proxy"  # the proxy's container: squid, which lets through Claude's API alone

# The steps of a turn are in program.md, the brief you edit. Only the IDEA line is asked for
# here, because turn() reads it.
# TODO: How much better on the practice year is "clearly better" (program.md, step 6)? No fixed
# number fits every idea; a practice.py running the judge's reshuffle test could answer it.
PROMPT = ("Read program.md and follow its steps. "
          "Your last line must be: IDEA: <the idea you tried, in under 12 words>")
AGENT = ["timeout", "1800",                         # a turn still running after 30 minutes is stopped
         "claude", "-p", PROMPT,
         "--model", "sonnet",
         "--permission-mode", "acceptEdits",            # may edit files in its own workspace only
         "--allowedTools", "Bash(python *),Bash(python3 *)",  # and run python; anything else is refused
         "--disallowedTools", "WebSearch,WebFetch",     # no web tools: its only sources are the lab files
         "--output-format", "json"]


def build() -> None:
    """Build the three Docker images. Quick when nothing has changed: Docker reuses its cache."""
    for name in IMAGES:
        subprocess.run(["docker", "build", "--quiet", "-t", f"toy-{name}", str(ROOT / "docker" / name)],
                       check=True, stdout=subprocess.DEVNULL)


def need_login_token() -> None:
    """Stop with instructions if there's no login token to pass into the agent sandbox.
    (Your Mac keeps its own Claude login in the Keychain, which a container can't use.)"""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        sys.exit("The agent sandbox needs a Claude login token. In your shell, run:\n"
                 "    claude setup-token\n"
                 "    export CLAUDE_CODE_OAUTH_TOKEN=<the token it prints>\n"
                 "then run this command again.")


def start_proxy() -> None:
    """Start the proxy: the only way out of the agent sandbox's network, and it leads to
    Claude's API alone. It keeps running after a run, and its log lists every request the
    agent sent through it (the next run or test starts a fresh proxy, with an empty log):
        docker exec toy-agent-proxy cat /var/log/squid/access.log"""
    subprocess.run(["docker", "rm", "--force", PROXY], capture_output=True)  # remove an old proxy, if any
    subprocess.run(["docker", "network", "rm", NETWORK], capture_output=True)  # and an old network, however it was made
    subprocess.run(["docker", "network", "create", "--internal", NETWORK],  # --internal: no way out of it but the proxy
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "run", "--detach", "--name", PROXY, "--network", NETWORK, "toy-agent-proxy"],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "network", "connect", "bridge", PROXY], check=True)  # the proxy's other side: the internet
    for _ in range(50):  # wait up to 5 seconds for squid to start listening (it takes about half a second)
        logs = subprocess.run(["docker", "logs", PROXY], capture_output=True, text=True)
        if "Accepting HTTP Socket connections" in logs.stdout + logs.stderr:
            return
        time.sleep(0.1)
    raise RuntimeError(f"the proxy didn't start; see: docker logs {PROXY}")


def agent_sandbox(work) -> list[str]:
    """The docker command that starts the agent sandbox, up to the image name. The sandbox
    sees only the turn's workspace, and its only way out is the proxy."""
    return ["docker", "run", "--rm", "--pull", "never",
            "--network", NETWORK,                                  # no internet, except through the proxy
            "-e", f"HTTPS_PROXY=http://{PROXY}:3128",              # so Claude Code sends its requests to the proxy
            "-e", "CLAUDE_CODE_OAUTH_TOKEN",                       # your login token, passed on by name from your shell
            "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",    # no update checks or error reports
            "-v", f"{work}:/work",                                 # the turn's workspace: the only folder from your Mac
            "toy-agent-sandbox"]


def copy_submission(work, candidate) -> bool:
    """Bring back the agent's model.py, but only if it's a plain file. A shortcut (symlink)
    would be followed here on your Mac, so it could point at any file."""
    submitted = work / "model.py"
    if submitted.is_symlink() or not submitted.is_file():
        return False
    shutil.copy(submitted, candidate)
    return True


def judge(*args) -> dict:
    run = subprocess.run([sys.executable, str(JUDGE / "judge.py"), *map(str, args)],
                         capture_output=True, text=True, timeout=900)
    if run.returncode != 0:
        lines = run.stderr.strip().splitlines()
        return {"error": lines[-1] if lines else "judge failed"}
    return json.loads(run.stdout)


def read_ledger() -> list[dict[str, str]]:
    rows = LEDGER.read_text().strip().splitlines()[1:]
    return [dict(zip(COLUMNS, r.split("\t"))) for r in rows]


def log(turn, idea, decision, val_mae="-", win_pct="-", bar_pct="-") -> None:
    """Add a row to the ledger (in COLUMNS order) and print it. Scores left out show as "-"."""
    row = [str(v).replace("\t", " ") for v in (turn, idea, val_mae, win_pct, bar_pct, decision)]
    with LEDGER.open("a") as f:
        f.write("\t".join(row) + "\n")
    print(f"turn {turn}: {idea} | validation MAE {val_mae} | won {win_pct}% (needed {bar_pct}%) | {decision}", flush=True)


def start() -> None:
    HISTORY.mkdir(exist_ok=True)
    shutil.copy(LAB / "model.py", HISTORY / "turn_0.py")
    LEDGER.write_text("\t".join(COLUMNS) + "\n")
    log(0, "baseline: average of the last 7 days", "start", val_mae=judge("score", "val", LAB / "model.py")["mae"])


def turn(t) -> None:
    """One turn of the loop: steps 1-4 at the top of this file."""
    # 1. copy the lab files into a fresh, empty workspace (a new temp folder)
    work = Path(tempfile.mkdtemp(prefix=f"agent-turn{t}-"))
    for name in AGENT_FILES:
        shutil.copy(LAB / name, work / name)

    # 2. the agent changes model.py in the agent sandbox, and ends its reply with "IDEA: ..."
    began = time.time()
    run = subprocess.run(agent_sandbox(work) + AGENT, capture_output=True, text=True)
    try:
        reply = json.loads(run.stdout)
    except json.JSONDecodeError:
        reply = {"result": run.stdout[-1000:] + run.stderr[-1000:]}
    found = re.findall(r"IDEA:\s*(.+)", reply.get("result") or "")
    idea = found[-1].strip().strip("*") if found else "(no description)"

    # 3. bring back model.py, delete the workspace, and save the turn in history/
    candidate = HISTORY / f"turn_{t}.py"
    submitted = copy_submission(work, candidate)  # only model.py comes back
    shutil.rmtree(work)
    (HISTORY / f"turn_{t}.json").write_text(json.dumps({
        "idea": idea, "seconds": round(time.time() - began),
        "cost_usd": reply.get("total_cost_usd"), "agent_reply": reply.get("result")}, indent=2))

    # skip the judge if the agent sent back nothing new
    if not submitted:
        log(t, idea, "no model.py")  # nothing to judge, so it doesn't count as a try
        return
    if candidate.read_text() == (LAB / "model.py").read_text():
        log(t, idea, "no change")  # nothing submitted, so it doesn't count as a try
        return

    # the judge compares it with the champion on the validation year; the bar rises with k
    k = 1 + sum(r["decision"] in TRIED for r in read_ledger())  # this is the k-th idea tested
    verdict = judge("compare", LAB / "model.py", candidate, k)

    # 4. keep it (new champion) or throw it away, and write the verdict into ledger.tsv
    if "error" in verdict:
        log(t, idea, "crashed")
        return
    if verdict["keep"]:
        shutil.copy(candidate, LAB / "model.py")
    log(t, idea, "KEPT" if verdict["keep"] else "thrown away",
        verdict["candidate_mae"], verdict["win_pct"], verdict["bar_pct"])


def holdout() -> None:
    """Score the baseline and every kept model on the holdout: the 6 months nobody has looked at."""
    print(f"{'model':<58}{'val':>6}{'holdout':>9}")
    for r in read_ledger():
        if r["decision"] in ("start", "KEPT"):
            holdout_mae = judge("score", "holdout", HISTORY / f"turn_{r['turn']}.py")["mae"]
            print(f"{'turn ' + r['turn'] + ': ' + r['idea']:<58.58}{float(r['val_mae']):>6.2f}{holdout_mae:>9.2f}")
    key = JUDGE / "answer_key.py"
    val_mae, holdout_mae = judge("score", "val", key)["mae"], judge("score", "holdout", key)["mae"]
    print(f"{'answer key (the exact recipe; about the best possible)':<58}{val_mae:>6.2f}{holdout_mae:>9.2f}")


def reset() -> None:
    shutil.copy(HISTORY / "turn_0.py", LAB / "model.py")
    shutil.rmtree(HISTORY)
    LEDGER.unlink()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "build":
        build()
    elif cmd == "run":
        need_login_token()
        build()
        start_proxy()
        if not LEDGER.exists():
            start()
        last = int(read_ledger()[-1]["turn"])
        for t in range(last + 1, last + 1 + int(sys.argv[2] if len(sys.argv) > 2 else 5)):
            turn(t)
    elif cmd == "holdout":
        build()
        holdout()
    elif cmd == "reset":
        reset()
