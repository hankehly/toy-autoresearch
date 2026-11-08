"""Proves that the isolation holds. Run it from the repo root:

    uv run scripts/test_sandboxes.py

The model sandbox: honest models must score exactly what the old judge gave them, and
models that try to cheat must crash instead of scoring.
The agent sandbox: its only way out is the proxy, which lets through Claude's API alone.
Only a plain model.py comes back from it, and a run without a login token stops at once.
Prints PASS or FAIL for each check, and exits with 1 if any check failed.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "judge")]
import judge  # noqa: E402  (imported after the path is set: these files live in the repo, not here)
import loop  # noqa: E402

loop.build()  # test the sandboxes as the current files describe them
TMP = Path(tempfile.mkdtemp(prefix="test-sandboxes-"))
failures = 0


def check(name, passed, detail="") -> None:
    global failures
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def write_model(name, code) -> Path:
    """Save a test model as <TMP>/<name>/model.py and return its path."""
    path = TMP / name / "model.py"
    path.parent.mkdir()
    path.write_text(code)
    return path


def val_score(model_path) -> float | Exception:
    """The model's average miss on the validation year, or the error that stopped it."""
    try:
        return round(float(judge.daily_misses(model_path, "val").mean()), 3)
    except Exception as error:
        return error


# --- The model sandbox: honest models score exactly as before ----------------------
# The scores are what the old judge gave these models, before the sandboxes existed.

SEVEN_DAY_AVERAGE_THAT_PRINTS = '''
def fit(history) -> None:
    print("fitting")  # a print must never be mistaken for the model's reply
    return None


def predict(state, history, today) -> float:
    return history["load"].tail(7).mean()
'''

MONTH_AVERAGE_LEARNED_IN_FIT = '''
def fit(history) -> float:
    return history["load"].tail(28).mean()


def predict(state, history, today) -> float:
    return state
'''

honest = [
    ("7-day average that prints", write_model("seven", SEVEN_DAY_AVERAGE_THAT_PRINTS), 5.782),
    ("28-day average learned in fit", write_model("month", MONTH_AVERAGE_LEARNED_IN_FIT), 7.528),
    ("answer key", ROOT / "judge" / "answer_key.py", 2.830),
]
for name, path, old_score in honest:
    score = val_score(path)
    check(f"scores as before: {name}", score == old_score, f"got {score!r}, the old judge gave {old_score}")


# --- The model sandbox: cheats crash instead of scoring -----------------------------
# A crash in the sandbox reaches the judge as RuntimeError. Any other result means the
# cheat ran somewhere it could do harm.

cheats = {
    "peeks at the judge's variables": '''
import inspect


def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return inspect.currentframe().f_back.f_locals["day"].load  # today's answer, if it's there
''',
    "reads the answer file": f'''
import pandas as pd

ANSWERS = pd.read_csv("{ROOT / "judge" / "full_data.csv"}", parse_dates=["date"]).set_index("date")["load"]


def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return ANSWERS[today["date"]]
''',
    "reaches the internet": '''
import urllib.request


def fit(history) -> None:
    urllib.request.urlopen("https://example.com", timeout=5)
    return None


def predict(state, history, today) -> float:
    return 100.0
''',
    "leaves a note for the next agent": '''
def fit(history) -> None:
    with open(__file__, "a") as f:  # model.py is the one file the sandbox shares with your Mac
        f.write(f"# hidden days seen so far: {len(history)}\\n")
    return None


def predict(state, history, today) -> float:
    return 100.0
''',
}
for i, (name, code) in enumerate(cheats.items()):
    result = val_score(write_model(f"cheat{i}", code))
    check(f"cheat is stopped: {name}", isinstance(result, RuntimeError), f"got {result!r}")


# --- The agent sandbox: only a plain model.py comes back ----------------------------

handoff = TMP / "handoff"
handoff.mkdir()
kept = TMP / "kept.py"
check("hand-off: no model.py, nothing comes back", loop.copy_submission(handoff, kept) is False)

(TMP / "elsewhere.txt").write_text("some other file on your Mac")
(handoff / "model.py").symlink_to(TMP / "elsewhere.txt")
check("hand-off: a shortcut is refused", loop.copy_submission(handoff, kept) is False and not kept.exists())

(handoff / "model.py").unlink()
(handoff / "model.py").write_text("# a plain model.py\n")
check("hand-off: a plain model.py comes back",
      loop.copy_submission(handoff, kept) is True and kept.read_text() == "# a plain model.py\n")


# --- The agent sandbox: no login token, no run --------------------------------------
# Checked in a separate Python, so the test never starts a real agent turn.

no_token = {name: value for name, value in os.environ.items() if name != "CLAUDE_CODE_OAUTH_TOKEN"}
stop = subprocess.run([sys.executable, "-c", "import loop; loop.need_login_token()"],
                      cwd=ROOT, env=no_token, capture_output=True, text=True)
check("without a login token, run stops and says how to get one",
      stop.returncode != 0 and "claude setup-token" in stop.stderr, " ".join(stop.stderr.split())[-110:])


# --- The agent sandbox: its only way out is the proxy, open to Claude's API alone ----
# The probe runs inside the very sandbox the agent gets: loop.agent_sandbox().

# A leftover network with a way out (made by hand, say) must not survive start_proxy().
subprocess.run(["docker", "rm", "--force", loop.PROXY], capture_output=True)
subprocess.run(["docker", "network", "rm", loop.NETWORK], capture_output=True)
subprocess.run(["docker", "network", "create", loop.NETWORK], check=True, stdout=subprocess.DEVNULL)

loop.start_proxy()
loop.start_proxy()  # a proxy left over from an earlier run is replaced, not tripped over

PROBE = """
import json, os, subprocess, sys, urllib.error, urllib.request


def reach(url, through_proxy=True) -> str:
    no_proxy = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener() if through_proxy else urllib.request.build_opener(no_proxy)
    try:
        opener.open(url, timeout=10)
    except urllib.error.HTTPError:
        pass  # an error page is still an answer: the site was reached
    except Exception as error:
        return str(error)
    return "answered"


print(json.dumps({
    "claude": reach("https://api.anthropic.com/"),
    "github": reach("https://github.com/"),
    "github_around_the_proxy": reach("https://github.com/", through_proxy=False),
    "sees_the_repo": os.path.exists(sys.argv[1]),
    "claude_version": subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout,
}))
"""
work = TMP / "agent-workspace"
work.mkdir()
probe = subprocess.run(loop.agent_sandbox(work) + ["python", "-c", PROBE, str(ROOT)], capture_output=True, text=True)
seen = json.loads(probe.stdout or "{}")
check("agent sandbox reaches Claude's API through the proxy", seen.get("claude") == "answered", seen.get("claude"))
check("agent sandbox can't reach GitHub through the proxy", "403" in seen.get("github", ""), seen.get("github"))
check("agent sandbox can't go around the proxy",
      seen.get("github_around_the_proxy", "answered") != "answered", seen.get("github_around_the_proxy"))
check("agent sandbox can't see the repo on your Mac", seen.get("sees_the_repo") is False, seen.get("sees_the_repo"))
check("agent sandbox has the Mac's Claude Code version",
      "2.1.319" in seen.get("claude_version", ""), (seen.get("claude_version") or probe.stderr[-200:]).strip())


shutil.rmtree(TMP)
print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
sys.exit(1 if failures else 0)
