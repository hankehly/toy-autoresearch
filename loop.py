"""The research loop. Each turn:
  1. copy lab/ into a fresh, empty workspace (the agent gets no path to the judge or the vault)
  2. a fresh Claude Code session reads program.md + ledger.tsv and changes model.py
  3. only model.py comes back; the judge compares it with the champion on the dev months,
     with a bar that rises with every idea tested (see judge.py)
  4. keep it (new champion) or throw it away, and write the verdict into ledger.tsv

    python loop.py run 5     # five more turns (continues from the ledger)
    python loop.py vault     # open the vault once, at the end: Jul - Dec 2025
    python loop.py reset     # back to the baseline
    python loop.py build     # build the rooms' Docker images (run and vault do this first anyway)
"""
import json
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
ROOMS = ["model-room"]  # folders in rooms/, each built as the Docker image toy-<name>

PROMPT = ("Read program.md, then ledger.tsv, then model.py. Make your one change to model.py, "
          "run check.py with python, and fix model.py if it crashes. "
          "Your last line must be: IDEA: <your change in under 12 words>")
AGENT = ["claude", "-p", PROMPT,
         "--model", "sonnet",
         "--permission-mode", "acceptEdits",            # may edit files in its own workspace only
         "--allowedTools", "Bash(python *),Bash(python3 *)",  # and run python; anything else is refused
         "--output-format", "json"]


def build():
    """Build the rooms' Docker images. Quick when nothing has changed: Docker reuses its cache."""
    for name in ROOMS:
        subprocess.run(["docker", "build", "--quiet", "-t", f"toy-{name}", str(ROOT / "rooms" / name)],
                       check=True, stdout=subprocess.DEVNULL)


def judge(*args):
    run = subprocess.run([sys.executable, str(JUDGE / "judge.py"), *map(str, args)],
                         capture_output=True, text=True, timeout=900)
    if run.returncode != 0:
        lines = run.stderr.strip().splitlines()
        return {"error": lines[-1] if lines else "judge failed"}
    return json.loads(run.stdout)


COLUMNS = ["turn", "idea", "dev_mae", "win_pct", "bar_pct", "decision"]


def read_ledger():
    rows = LEDGER.read_text().strip().splitlines()[1:]
    return [dict(zip(COLUMNS, r.split("\t"))) for r in rows]


def log(turn, idea, dev_mae, win_pct, bar_pct, decision):
    with LEDGER.open("a") as f:
        f.write(f"{turn}\t{idea}\t{dev_mae}\t{win_pct}\t{bar_pct}\t{decision}\n")
    print(f"turn {turn}: {idea} | dev MAE {dev_mae} | won {win_pct}% (needed {bar_pct}%) | {decision}", flush=True)


def start():
    HISTORY.mkdir(exist_ok=True)
    shutil.copy(LAB / "model.py", HISTORY / "turn_0.py")
    LEDGER.write_text("\t".join(COLUMNS) + "\n")
    log(0, "baseline: average of the last 7 days", judge("score", "dev", LAB / "model.py")["mae"], "-", "-", "start")


def turn(t):
    work = Path(tempfile.mkdtemp(prefix=f"agent-turn{t}-"))
    for name in AGENT_FILES:
        shutil.copy(LAB / name, work / name)
    began = time.time()
    run = subprocess.run(AGENT, cwd=work, capture_output=True, text=True, timeout=1800)
    try:
        reply = json.loads(run.stdout)
    except json.JSONDecodeError:
        reply = {"result": run.stdout[-1000:] + run.stderr[-1000:]}
    found = re.findall(r"IDEA:\s*(.+)", reply.get("result") or "")
    idea = found[-1].strip().strip("*").replace("\t", " ") if found else "(no description)"
    candidate = HISTORY / f"turn_{t}.py"
    shutil.copy(work / "model.py", candidate)  # only model.py comes back
    shutil.rmtree(work)
    (HISTORY / f"turn_{t}.json").write_text(json.dumps({
        "idea": idea, "seconds": round(time.time() - began),
        "cost_usd": reply.get("total_cost_usd"), "agent_reply": reply.get("result")}, indent=2))

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


def vault():
    """Score the baseline and every kept model on the 6 months nobody has looked at."""
    print(f"{'model':<58}{'dev':>6}{'vault':>7}")
    for r in read_ledger():
        if r["decision"] in ("start", "KEPT"):
            v = judge("score", "vault", HISTORY / f"turn_{r['turn']}.py")["mae"]
            print(f"{'turn ' + r['turn'] + ': ' + r['idea']:<58.58}{float(r['dev_mae']):>6.2f}{v:>7.2f}")
    key = JUDGE / "answer_key.py"
    d, v = judge("score", "dev", key)["mae"], judge("score", "vault", key)["mae"]
    print(f"{'answer key (the exact recipe; about the best possible)':<58}{d:>6.2f}{v:>7.2f}")


def reset():
    shutil.copy(HISTORY / "turn_0.py", LAB / "model.py")
    shutil.rmtree(HISTORY)
    LEDGER.unlink()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "build":
        build()
    elif cmd == "run":
        build()
        if not LEDGER.exists():
            start()
        last = int(read_ledger()[-1]["turn"])
        for t in range(last + 1, last + 1 + int(sys.argv[2] if len(sys.argv) > 2 else 5)):
            turn(t)
    elif cmd == "vault":
        build()
        vault()
    elif cmd == "reset":
        reset()
