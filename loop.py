"""The research loop. Each turn:
  1. copy lab/ into a fresh, empty workspace
  2. a fresh Claude Code session reads program.md + ledger.tsv and changes model.py. It works
     in the agent sandbox: a container that sees only that workspace and reaches only Claude's API.
     It finishes with a structured reply: its idea, a lesson, its practice numbers, submit or not
  3. only model.py comes back, and the loop measures it against the champion on the practice
     year itself (the judge does the measuring, in the model sandbox)
  4. if the agent submitted it, the judge compares it with the champion on the validation
     months, with a bar that rises with every idea tested (see judge.py)
  5. keep it (new champion) or throw it away, and write the verdict into both ledgers

    uv run loop.py run 5     # five more turns (continues from the ledger)
    uv run loop.py holdout   # score the holdout once, at the end: Jul - Dec 2025
    uv run loop.py reset     # back to the baseline
    uv run loop.py build     # build the Docker images (run and holdout do this first anyway)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jsonschema

ROOT = Path(__file__).parent.resolve()
LAB, JUDGE, HISTORY = ROOT / "lab", ROOT / "judge", ROOT / "history"

# Two ledgers. The full one holds every score and is never copied into the agent's workspace.
# The agent's copy is rewritten from it after every turn, without the validation columns: an
# agent that could read validation scores could pick and combine near-misses by them, which
# is tuning on the judge's year. Practice scores are safe to show: the agent has that year.
LEDGER = ROOT / "ledger.tsv"
AGENT_LEDGER = LAB / "ledger.tsv"
COLUMNS = ["turn", "idea", "practice_champion", "practice", "practice_wins",
           "val_mae", "win_pct", "bar_pct", "verdict", "lesson", "reply_check"]
AGENT_COLUMNS = ["turn", "idea", "practice_champion", "practice", "practice_wins", "verdict", "lesson"]
TRIED = ("KEPT", "thrown away")  # the verdicts that count as a try, so they raise the bar
CHAMPIONS = ("start", "KEPT")    # the verdicts whose model became the champion
AGENT_FILES = ["program.md", "model.py", "check.py", "practice.py", "data.csv", "ledger.tsv"]
IMAGES = ["model-sandbox", "agent-sandbox", "agent-proxy"]  # folders in docker/, each built as the Docker image toy-<name>
NETWORK = "toy-agent-net"  # the agent sandbox's network: its only way out is the proxy
PROXY = "toy-agent-proxy"  # the proxy's container: squid, which lets through Claude's API alone

# The agent's reply: a JSON object in this shape, not free text. Claude Code gets the schema
# through --json-schema and returns the object as "structured_output"; turn() checks the
# object against the same schema again, because a run can end without a valid one.
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "idea": {"type": "string", "minLength": 3, "maxLength": 100,
                 "description": "The idea you tried, in under 12 words"},
        "lesson": {"type": "string", "minLength": 3, "maxLength": 300,
                   "description": "One sentence: what this turn taught you"},
        "practice": {
            "description": "What practice.py printed for champion.py vs model.py, or null if you couldn't run it",
            "anyOf": [
                {"type": "object",
                 "properties": {
                     "champion_mae": {"type": "number", "exclusiveMinimum": 0},
                     "model_mae": {"type": "number", "exclusiveMinimum": 0},
                     "wins_pct": {"type": "number", "minimum": 0, "maximum": 100},
                 },
                 "required": ["champion_mae", "model_mae", "wins_pct"],
                 "additionalProperties": False},
                {"type": "null"},
            ],
        },
        "submit": {"type": "boolean",
                   "description": "true sends model.py to the judge; false logs it as dropped on practice"},
    },
    "required": ["idea", "lesson", "practice", "submit"],
    "additionalProperties": False,
}

# The steps of a turn are in program.md, the brief you edit.
# TODO: program.md tells the agent to submit only if model.py wins at least 90% of the practice
# reshuffles. 90% is a starting guideline, and the loop doesn't enforce it. Revisit it once
# the full ledger has a few dozen judged ideas: compare practice_wins with the verdicts.
PROMPT = "Read program.md and follow its steps. Finish with the structured reply."
AGENT = ["timeout", "1800",                         # a turn still running after 30 minutes is stopped
         "claude", "-p", PROMPT,
         "--model", "sonnet",
         "--permission-mode", "acceptEdits",            # may edit files in its own workspace only
         "--allowedTools", "Bash(python *),Bash(python3 *)",  # and run python; anything else is refused
         "--disallowedTools", "WebSearch,WebFetch",     # no web tools: its only sources are the lab files
         "--output-format", "json",
         "--json-schema", json.dumps(REPLY_SCHEMA)]     # the reply's shape; it comes back as "structured_output"


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


def run_agent(work) -> dict:
    """Run one agent session in the agent sandbox, on the workspace. Returns what Claude Code
    printed: a JSON object with the agent's "structured_output", its "result" text and the cost."""
    run = subprocess.run(agent_sandbox(work) + AGENT, capture_output=True, text=True)
    try:
        return json.loads(run.stdout)
    except json.JSONDecodeError:
        return {"result": run.stdout[-1000:] + run.stderr[-1000:]}


def check_reply(reported, measured) -> str:
    """Did the agent decide on the same practice numbers the ledger records? reported is the
    "practice" part of its reply; measured is the loop's own measurement (None if there is none).
    The loop's numbers are the record either way, so this never changes a verdict."""
    if reported is None:
        return "not run"  # the agent says it couldn't run practice.py
    if measured is None:
        return "-"        # nothing to compare with
    gaps = (abs(reported["champion_mae"] - measured["champion_mae"]),  # in MWh
            abs(reported["model_mae"] - measured["candidate_mae"]))
    wins_gap = abs(reported["wins_pct"] - measured["win_pct"])         # in points
    # rounded, so floating-point error can't push a gap of exactly 0.001 over the line
    if round(max(gaps), 6) > 0.001 or round(wins_gap, 6) > 0.1:
        return "practice mismatch"
    return "ok"


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
    """The rows of the full ledger."""
    rows = LEDGER.read_text().strip().splitlines()[1:]
    return [dict(zip(COLUMNS, r.split("\t"))) for r in rows]


def ideas_judged() -> int:
    """How many ideas the judge has tested so far. The next one is tested against a higher bar."""
    return sum(r["verdict"] in TRIED for r in read_ledger())


def log(turn, idea, verdict, **cells) -> None:
    """Add a row to the full ledger and print it, then rewrite the agent's copy.
    cells are the row's other columns by name; those left out show as "-"."""
    cells = {"turn": turn, "idea": idea, "verdict": verdict, **cells}
    row = {c: " ".join(str(cells.get(c, "-")).split()) or "-" for c in COLUMNS}  # one line of text per cell, no tabs
    with LEDGER.open("a") as f:
        f.write("\t".join(row.values()) + "\n")
    print(f"turn {turn}: {row['idea']} | practice {row['practice_champion']} -> {row['practice']}, won {row['practice_wins']}%"
          f" | validation MAE {row['val_mae']}, won {row['win_pct']}% (needed {row['bar_pct']}%) | {verdict}", flush=True)
    write_agent_ledger()


def write_agent_ledger() -> None:
    """Rewrite the agent's copy from the full ledger: practice numbers, verdicts and lessons,
    and no number measured on the validation year. The two comment lines on top say what the
    agent is up against: the champion, and the bar the next idea must clear."""
    rows = read_ledger()
    champion = [r for r in rows if r["verdict"] in CHAMPIONS][-1]
    bar_pct = judge("bar", ideas_judged() + 1)["bar_pct"]
    lines = [f"# Champion: turn {champion['turn']}, practice {champion['practice']} MWh",
             f"# Ideas judged: {ideas_judged()}. The next must win {bar_pct}% of the judge's reshuffles.",
             "\t".join(AGENT_COLUMNS)]
    lines += ["\t".join(r[c] for c in AGENT_COLUMNS) for r in rows]
    AGENT_LEDGER.write_text("\n".join(lines) + "\n")


def start() -> None:
    HISTORY.mkdir(exist_ok=True)
    shutil.copy(LAB / "model.py", HISTORY / "turn_0.py")
    LEDGER.write_text("\t".join(COLUMNS) + "\n")
    log(0, "baseline: average of the last 7 days", "start",
        practice=judge("score", "practice", LAB / "model.py")["mae"],
        val_mae=judge("score", "val", LAB / "model.py")["mae"])


def turn(t) -> None:
    """One turn of the loop: steps 1-5 at the top of this file."""
    # 1. copy the lab files into a fresh, empty workspace (a new temp folder)
    champion = LAB / "model.py"
    work = Path(tempfile.mkdtemp(prefix=f"agent-turn{t}-"))
    for name in AGENT_FILES:
        shutil.copy(LAB / name, work / name)

    # 2. the agent changes model.py in the agent sandbox, and finishes with a structured reply
    began = time.time()
    result = run_agent(work)
    reply = result.get("structured_output")  # missing if the run ended without a valid one
    valid = jsonschema.Draft202012Validator(REPLY_SCHEMA).is_valid(reply)

    # 3. bring back model.py, delete the workspace, and save the turn in history/
    candidate = HISTORY / f"turn_{t}.py"
    if not copy_submission(work, candidate):  # only model.py comes back
        shutil.copy(champion, candidate)      # no plain model.py: the same as changing nothing
    shutil.rmtree(work)
    (HISTORY / f"turn_{t}.json").write_text(json.dumps({
        "seconds": round(time.time() - began), "cost_usd": result.get("total_cost_usd"),
        "structured_output": reply, "agent_reply": result.get("result")}, indent=2))

    # the loop measures practice itself, so the ledger never depends on the agent's own numbers
    changed = candidate.read_bytes() != champion.read_bytes()
    measured, crashed = None, False  # the loop's practice numbers; None if model.py didn't change or didn't run
    if changed:
        measured = judge("practice", champion, candidate)
        if "error" in measured:
            measured, crashed = None, True
    row = {}  # the row's cells so far
    if measured:
        row = {"practice_champion": measured["champion_mae"], "practice": measured["candidate_mae"],
               "practice_wins": measured["win_pct"]}

    # the verdicts that never reach the judge. None of them counts as a try, so the bar stays put
    if not valid:
        log(t, "-", "invalid reply", **row)  # the practice numbers, if any, are still logged
        return
    idea = reply["idea"]
    row["lesson"] = reply["lesson"]
    row["reply_check"] = check_reply(reply["practice"], measured)
    if not changed:
        log(t, idea, "no change", **row)
        return
    if crashed:
        log(t, idea, "crashed", **row)  # only the word: a model's error text could carry anything
        return
    if not reply["submit"]:
        log(t, idea, "dropped on practice", **row)
        return

    # 4. the judge compares it with the champion on the validation year; the bar rises with k
    k = 1 + ideas_judged()  # this is the k-th idea tested
    judged = judge("compare", champion, candidate, k)

    # 5. keep it (new champion) or throw it away, and write the verdict into both ledgers
    if "error" in judged:
        log(t, idea, "crashed", **row)
        return
    if judged["keep"]:
        shutil.copy(candidate, champion)
    log(t, idea, "KEPT" if judged["keep"] else "thrown away",
        val_mae=judged["candidate_mae"], win_pct=judged["win_pct"], bar_pct=judged["bar_pct"], **row)


def holdout() -> None:
    """Score the baseline and every kept model on the holdout: the 6 months nobody has looked at."""
    print(f"{'model':<58}{'val':>6}{'holdout':>9}")
    for r in read_ledger():
        if r["verdict"] in CHAMPIONS:
            holdout_mae = judge("score", "holdout", HISTORY / f"turn_{r['turn']}.py")["mae"]
            print(f"{'turn ' + r['turn'] + ': ' + r['idea']:<58.58}{float(r['val_mae']):>6.2f}{holdout_mae:>9.2f}")
    key = JUDGE / "answer_key.py"
    val_mae, holdout_mae = judge("score", "val", key)["mae"], judge("score", "holdout", key)["mae"]
    print(f"{'answer key (the exact recipe; about the best possible)':<58}{val_mae:>6.2f}{holdout_mae:>9.2f}")


def reset() -> None:
    shutil.copy(HISTORY / "turn_0.py", LAB / "model.py")
    shutil.rmtree(HISTORY)
    LEDGER.unlink()
    AGENT_LEDGER.unlink()


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
