"""The research loop. Each turn:
  1. copy lab/ into a fresh, empty workspace
  2. a fresh Claude Code session reads program.md + ledger.tsv and changes model.py. It works
     in the agent room: a container that sees only that workspace and reaches only Claude's API
  3. only model.py comes back; the judge compares it with the champion on the validation months,
     with a bar that rises with every idea tested (see judge.py)
  4. keep it (new champion) or throw it away, and write the verdict into ledger.tsv

    python loop.py run 5     # five more turns (continues from the ledger)
    python loop.py holdout   # score the holdout once, at the end: Jul - Dec 2025
    python loop.py reset     # back to the baseline
    python loop.py build     # build the rooms' Docker images (run and holdout do this first anyway)
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
LEDGER = LAB / "ledger.tsv"
AGENT_FILES = ["program.md", "model.py", "check.py", "data.csv", "ledger.tsv"]
ROOMS = ["model-room", "agent-room", "agent-door"]  # folders in rooms/, each built as the Docker image toy-<name>
NETWORK = "toy-agent-net"  # the agent room's network: its only way out is the door
DOOR = "toy-agent-door"    # the door's container, a web proxy that lets through Claude's API alone

PROMPT = ("Read program.md, then ledger.tsv, then model.py. Make your one change to model.py, "
          "run check.py with python, and fix model.py if it crashes. "
          "Your last line must be: IDEA: <your change in under 12 words>")
AGENT = ["timeout", "1800",                         # a turn still running after 30 minutes is stopped
         "claude", "-p", PROMPT,
         "--model", "sonnet",
         "--permission-mode", "acceptEdits",            # may edit files in its own workspace only
         "--allowedTools", "Bash(python *),Bash(python3 *)",  # and run python; anything else is refused
         "--disallowedTools", "WebSearch,WebFetch",     # no web tools: its only sources are the lab files
         "--output-format", "json"]


def build():
    """Build the rooms' Docker images. Quick when nothing has changed: Docker reuses its cache."""
    for name in ROOMS:
        subprocess.run(["docker", "build", "--quiet", "-t", f"toy-{name}", str(ROOT / "rooms" / name)],
                       check=True, stdout=subprocess.DEVNULL)


def need_login_token():
    """Stop with instructions if there's no login token to pass into the agent room.
    (Your Mac keeps its own Claude login in the Keychain, which a container can't use.)"""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        sys.exit("The agent room needs a Claude login token. In your shell, run:\n"
                 "    claude setup-token\n"
                 "    export CLAUDE_CODE_OAUTH_TOKEN=<the token it prints>\n"
                 "then run this command again.")


def open_door():
    """Open the door: the only way out of the agent room's network, and it leads to Claude's
    API alone. It stays open after a run, and its log lists every request the agent sent
    through it (the next run or test starts a fresh door, with an empty log):
        docker exec toy-agent-door cat /var/log/squid/access.log"""
    subprocess.run(["docker", "rm", "--force", DOOR], capture_output=True)  # remove an old door, if any
    subprocess.run(["docker", "network", "rm", NETWORK], capture_output=True)  # and an old network, however it was made
    subprocess.run(["docker", "network", "create", "--internal", NETWORK],  # --internal: no way out of it but the door
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "run", "--detach", "--name", DOOR, "--network", NETWORK, "toy-agent-door"],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "network", "connect", "bridge", DOOR], check=True)  # the door's other side: the internet
    for _ in range(50):  # wait up to 5 seconds for squid to start listening (it takes about half a second)
        logs = subprocess.run(["docker", "logs", DOOR], capture_output=True, text=True)
        if "Accepting HTTP Socket connections" in logs.stdout + logs.stderr:
            return
        time.sleep(0.1)
    raise RuntimeError(f"the door didn't open; see: docker logs {DOOR}")


def agent_room(work):
    """The docker command that starts the agent room, up to the image name. The room sees only
    the turn's workspace, and its only way out is the door."""
    return ["docker", "run", "--rm", "--pull", "never",
            "--network", NETWORK,                                  # no internet, except through the door
            "-e", f"HTTPS_PROXY=http://{DOOR}:3128",               # so Claude Code sends its requests to the door
            "-e", "CLAUDE_CODE_OAUTH_TOKEN",                       # your login token, passed on by name from your shell
            "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",    # no update checks or error reports
            "-v", f"{work}:/work",                                 # the turn's workspace: the only folder from your Mac
            "toy-agent-room"]


def copy_submission(work, candidate):
    """Bring back the agent's model.py, but only if it's a plain file. A shortcut (symlink)
    would be followed here on your Mac, so it could point at any file."""
    submitted = work / "model.py"
    if submitted.is_symlink() or not submitted.is_file():
        return False
    shutil.copy(submitted, candidate)
    return True


def judge(*args):
    run = subprocess.run([sys.executable, str(JUDGE / "judge.py"), *map(str, args)],
                         capture_output=True, text=True, timeout=900)
    if run.returncode != 0:
        lines = run.stderr.strip().splitlines()
        return {"error": lines[-1] if lines else "judge failed"}
    return json.loads(run.stdout)


COLUMNS = ["turn", "idea", "val_mae", "win_pct", "bar_pct", "decision"]


def read_ledger():
    rows = LEDGER.read_text().strip().splitlines()[1:]
    return [dict(zip(COLUMNS, r.split("\t"))) for r in rows]


def log(turn, idea, val_mae, win_pct, bar_pct, decision):
    with LEDGER.open("a") as f:
        f.write(f"{turn}\t{idea}\t{val_mae}\t{win_pct}\t{bar_pct}\t{decision}\n")
    print(f"turn {turn}: {idea} | validation MAE {val_mae} | won {win_pct}% (needed {bar_pct}%) | {decision}", flush=True)


def start():
    HISTORY.mkdir(exist_ok=True)
    shutil.copy(LAB / "model.py", HISTORY / "turn_0.py")
    LEDGER.write_text("\t".join(COLUMNS) + "\n")
    log(0, "baseline: average of the last 7 days", judge("score", "val", LAB / "model.py")["mae"], "-", "-", "start")


def turn(t):
    work = Path(tempfile.mkdtemp(prefix=f"agent-turn{t}-"))
    for name in AGENT_FILES:
        shutil.copy(LAB / name, work / name)
    began = time.time()
    run = subprocess.run(agent_room(work) + AGENT, capture_output=True, text=True)
    try:
        reply = json.loads(run.stdout)
    except json.JSONDecodeError:
        reply = {"result": run.stdout[-1000:] + run.stderr[-1000:]}
    found = re.findall(r"IDEA:\s*(.+)", reply.get("result") or "")
    idea = found[-1].strip().strip("*").replace("\t", " ") if found else "(no description)"
    candidate = HISTORY / f"turn_{t}.py"
    submitted = copy_submission(work, candidate)  # only model.py comes back
    shutil.rmtree(work)
    (HISTORY / f"turn_{t}.json").write_text(json.dumps({
        "idea": idea, "seconds": round(time.time() - began),
        "cost_usd": reply.get("total_cost_usd"), "agent_reply": reply.get("result")}, indent=2))

    if not submitted:
        log(t, idea, "-", "-", "-", "no model.py")  # nothing to judge, so it doesn't count as a try
        return
    if candidate.read_text() == (LAB / "model.py").read_text():
        log(t, idea, "-", "-", "-", "no change")  # nothing submitted, so it doesn't count as a try
        return
    k = 1 + sum(r["decision"] in ("KEPT", "thrown away") for r in read_ledger())  # this is the k-th idea tested
    verdict = judge("compare", LAB / "model.py", candidate, k)
    if "error" in verdict:
        log(t, idea, "-", "-", "-", "crashed")
    elif verdict["keep"]:
        shutil.copy(candidate, LAB / "model.py")
        log(t, idea, verdict["candidate_mae"], verdict["win_pct"], verdict["bar_pct"], "KEPT")
    else:
        log(t, idea, verdict["candidate_mae"], verdict["win_pct"], verdict["bar_pct"], "thrown away")


def holdout():
    """Score the baseline and every kept model on the holdout: the 6 months nobody has looked at."""
    print(f"{'model':<58}{'val':>6}{'holdout':>9}")
    for r in read_ledger():
        if r["decision"] in ("start", "KEPT"):
            holdout_mae = judge("score", "holdout", HISTORY / f"turn_{r['turn']}.py")["mae"]
            print(f"{'turn ' + r['turn'] + ': ' + r['idea']:<58.58}{float(r['val_mae']):>6.2f}{holdout_mae:>9.2f}")
    key = JUDGE / "answer_key.py"
    val_mae, holdout_mae = judge("score", "val", key)["mae"], judge("score", "holdout", key)["mae"]
    print(f"{'answer key (the exact recipe; about the best possible)':<58}{val_mae:>6.2f}{holdout_mae:>9.2f}")


def reset():
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
        open_door()
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
