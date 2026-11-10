"""Proves that a turn of the loop is logged the way the ledgers promise. Run it from the repo root:

    uv run scripts/test_loop.py

Every turn here is a real loop.turn() on a temporary copy of lab/, with the real judge and
the real model sandbox. Only the agent is replaced: a stand-in that changes model.py and
returns a reply, as Claude Code would. So no login token is needed, and nothing in lab/,
history/ or the ledgers is touched.
Prints PASS or FAIL for each check, and exits with 1 if any check failed (about 1 minute).
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import loop  # noqa: E402  (imported after the path is set: loop.py lives in the repo root, not here)

loop.build()  # the judge runs every model in the model sandbox
TMP = Path(tempfile.mkdtemp(prefix="test-loop-"))
failures = 0


def check(name, passed, detail="") -> None:
    global failures
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


# --- Three models whose scores we know ----------------------------------------------
# They are turns 0, 1 and 3 of the example run. The expected numbers below come from that
# run's ledger (validation) and from issue #1's table (practice).

BASELINE = '''
def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return history["load"].tail(7).mean()
'''

REGRESSION = '''
import numpy as np
import pandas as pd


def _features(dates, temps) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    return np.column_stack(cols)


def fit(history) -> np.ndarray:
    X = _features(history["date"], history["temp_forecast"])
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    X = _features([today["date"]], [today["temp_forecast"]])
    return float(X[0] @ state)
'''

WITH_YESTERDAY = '''
import numpy as np
import pandas as pd


def _features(dates, temps, prev) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    p = np.asarray(prev, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    cols += [np.maximum(17 - p, 0), np.maximum(p - 21, 0)]
    return np.column_stack(cols)


def fit(history) -> np.ndarray:
    t = history["temp_forecast"].to_numpy(float)
    prev = np.r_[t[0], t[:-1]]
    X = _features(history["date"], t, prev)
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    prev = history["temp_forecast"].iloc[-1] if len(history) else today["temp_forecast"]
    X = _features([today["date"]], [today["temp_forecast"]], [prev])
    return float(X[0] @ state)
'''


def write_model(name, code) -> Path:
    """Save a test model as <TMP>/<name>.py and return its path."""
    path = TMP / f"{name}.py"
    path.write_text(code)
    return path


baseline, regression = write_model("baseline", BASELINE), write_model("regression", REGRESSION)


# --- The judge can score the practice year, in the model sandbox ---------------------

score = loop.judge("score", "practice", regression)
check("judge scores the practice year: the regression misses by 3.036", score.get("mae") == 3.036, score)

practice = loop.judge("practice", baseline, regression)
check("judge compares two models on the practice year, with no bar",
      practice == {"champion_mae": 5.656, "candidate_mae": 3.036, "win_pct": 100.0}, practice)

bar = loop.judge("bar", 7)
check("judge names the bar for the 7th idea: 99.29%", bar == {"bar_pct": 99.29}, bar)

still = loop.judge("compare", regression, write_model("yesterday", WITH_YESTERDAY), 3)
check("judge's validation verdict is unchanged (turn 3 of the example run)",
      still == {"champion_mae": 2.915, "candidate_mae": 2.893, "win_pct": 67.91, "bar_pct": 98.33, "keep": False}, still)


# --- lab/practice.py gives the agent the same practice numbers ----------------------
# It runs where the agent runs it: in the agent sandbox, here with no network at all.

def run_practice(champion_code, model_code) -> dict:
    """What practice.py prints for these two models, as {name: number}, or its error."""
    work = Path(tempfile.mkdtemp(prefix="practice-", dir=TMP))
    for name in ("practice.py", "data.csv"):
        shutil.copy(ROOT / "lab" / name, work / name)
    (work / "champion.py").write_text(champion_code)
    (work / "model.py").write_text(model_code)
    run = subprocess.run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "-v", f"{work}:/work",
                          "toy-agent-sandbox", "python", "practice.py", "champion.py", "model.py"],
                         capture_output=True, text=True)
    lines = [line.split() for line in run.stdout.splitlines() if ":" in line]
    return {words[0].rstrip(":"): float(words[1]) for words in lines} or {"error": run.stderr[-300:]}


printed = run_practice(BASELINE, REGRESSION)
check("practice.py prints 3.036 for turn 1's model, as the judge does",
      printed == {"champion_mae": 5.656, "model_mae": 3.036, "wins_pct": 100.0}, printed)

printed = run_practice(REGRESSION, WITH_YESTERDAY)
judged = loop.judge("practice", regression, TMP / "yesterday.py")
check("practice.py and the judge agree on a close call (turn 3: 3.036 -> 2.880)",
      printed == {"champion_mae": 3.036, "model_mae": 2.88, "wins_pct": judged.get("win_pct")}, f"{printed} vs {judged}")


# --- A temporary lab for the loop to work on -----------------------------------------
# The loop's folders are pointed at copies, so the real lab/, history/ and ledgers stay as they are.

loop.LAB, loop.HISTORY = TMP / "lab", TMP / "history"
loop.LEDGER, loop.AGENT_LEDGER = TMP / "ledger.tsv", TMP / "lab" / "ledger.tsv"
loop.LAB.mkdir()
for name in ("program.md", "check.py", "practice.py", "data.csv"):
    shutil.copy(ROOT / "lab" / name, loop.LAB / name)
(loop.LAB / "model.py").write_text(BASELINE)


def last_row() -> dict[str, str]:
    """The newest row of the full ledger."""
    return loop.read_ledger()[-1]


def cells(row, *names) -> list[str]:
    return [row[name] for name in names]


PRACTICE = ("practice_champion", "practice", "practice_wins")
VALIDATION = ("val_mae", "win_pct", "bar_pct")


# --- The two ledgers ------------------------------------------------------------------

loop.start()
row = last_row()
check("start: the full ledger has the columns issue #1 lists",
      list(row) == ["turn", "idea", "practice_champion", "practice", "practice_wins",
                    "val_mae", "win_pct", "bar_pct", "verdict", "lesson", "reply_check"], list(row))
check("start: the baseline row has its practice score and its validation score",
      cells(row, "turn", "practice", "val_mae", "verdict") == ["0", "5.656", "5.782", "start"], row)

agent_ledger = loop.AGENT_LEDGER.read_text().splitlines()
check("agent's copy: two comment lines name the champion and the next bar",
      agent_ledger[:2] == ["# Champion: turn 0, practice 5.656 MWh",
                           "# Ideas judged: 0. The next must win 95.0% of the judge's reshuffles."], agent_ledger[:2])
check("agent's copy: no validation column",
      agent_ledger[2].split("\t") == ["turn", "idea", "practice_champion", "practice", "practice_wins", "verdict", "lesson"],
      agent_ledger[2])
check("agent's copy: the baseline row, without its validation score",
      agent_ledger[3].split("\t") == ["0", "baseline: average of the last 7 days", "-", "5.656", "-", "start", "-"],
      agent_ledger[3])


# --- Turns, with a stand-in for the agent ---------------------------------------------

seen = {}  # what the stand-in agent found in its workspace on the latest turn
MISSING = "delete model.py"  # for play(): the stand-in agent leaves no model.py behind


def reply(practice, submit, idea="an idea worth trying", lesson="What this turn taught me.") -> dict:
    """A structured reply, as the agent would fill it in. practice is the three numbers
    (champion_mae, model_mae, wins_pct) it says practice.py printed, or None."""
    names = ("champion_mae", "model_mae", "wins_pct")
    return {"idea": idea, "lesson": lesson, "submit": submit,
            "practice": dict(zip(names, practice)) if practice else None}


def play(model_code, result) -> dict[str, str]:
    """Run the next turn. The stand-in agent writes model_code over model.py (None: it leaves
    model.py alone; MISSING: it deletes it) and Claude Code returns result.
    Returns the turn's row in the full ledger."""
    def stand_in(work) -> dict:
        seen["files"] = sorted(path.name for path in work.iterdir())
        seen["ledger"] = (work / "ledger.tsv").read_text()
        (work / "practice.py").write_text("print('champion_mae: 9.999')")  # tampering must change nothing
        if model_code == MISSING:
            (work / "model.py").unlink()
        elif model_code is not None:
            (work / "model.py").write_text(model_code)
        return result

    loop.run_agent = stand_in
    loop.turn(int(last_row()["turn"]) + 1)
    return last_row()


def agent_ledger_top() -> str:
    """The second comment line of the agent's copy: how many ideas were judged, and the next bar."""
    return loop.AGENT_LEDGER.read_text().splitlines()[1]


# Turn 1: a clear gain, submitted with the right practice numbers.
row = play(REGRESSION, {"result": "Done.", "total_cost_usd": 0.1,
                        "structured_output": reply((5.656, 3.036, 100.0), submit=True,
                                                   idea="weekday plus heating/cooling regression",
                                                   lesson="Weekdays and temperature explain most of the load.")})
check("kept: a submitted idea that clears the bar is KEPT", row["verdict"] == "KEPT", row["verdict"])
check("kept: the practice numbers are the loop's own", cells(row, *PRACTICE) == ["5.656", "3.036", "100.0"], cells(row, *PRACTICE))
check("kept: the full ledger has the validation numbers", cells(row, *VALIDATION) == ["2.915", "100.0", "95.0"], cells(row, *VALIDATION))
check("kept: the idea and the lesson come from the structured reply",
      cells(row, "idea", "lesson") == ["weekday plus heating/cooling regression", "Weekdays and temperature explain most of the load."], row)
check("kept: reply_check is ok when the agent's practice numbers match the loop's", row["reply_check"] == "ok", row["reply_check"])
check("kept: model.py is the new champion", (loop.LAB / "model.py").read_text() == REGRESSION)
check("kept: the agent's copy names the new champion and a higher bar",
      loop.AGENT_LEDGER.read_text().splitlines()[:2] == ["# Champion: turn 1, practice 3.036 MWh",
                                                         "# Ideas judged: 1. The next must win 97.5% of the judge's reshuffles."],
      loop.AGENT_LEDGER.read_text().splitlines()[:2])
check("workspace: the agent gets practice.py and nothing but the lab files",
      seen["files"] == ["check.py", "data.csv", "ledger.tsv", "model.py", "practice.py", "program.md"], seen["files"])
saved = json.loads((loop.HISTORY / "turn_1.json").read_text())
check("history: the turn's structured reply is saved", saved.get("structured_output", {}).get("submit") is True, saved)

# Turn 2: a changed model.py with submit false, and a practice number the agent got wrong.
row = play(WITH_YESTERDAY, {"structured_output": reply((3.036, 2.5, 100.0), submit=False)})
check("dropped: a changed model.py with submit false is 'dropped on practice'", row["verdict"] == "dropped on practice", row["verdict"])
check("dropped: its practice numbers are the loop's, not the agent's 2.5",
      cells(row, *PRACTICE) == ["3.036", "2.88", "99.9"], cells(row, *PRACTICE))
check("dropped: reply_check flags the agent's wrong number", row["reply_check"] == "practice mismatch", row["reply_check"])
check("dropped: it never reached the judge", cells(row, *VALIDATION) == ["-", "-", "-"], cells(row, *VALIDATION))
check("dropped: the bar doesn't rise", "Ideas judged: 1. The next must win 97.5%" in agent_ledger_top(), agent_ledger_top())
check("dropped: the champion stays, and the idea's code is saved in history/",
      (loop.LAB / "model.py").read_text() == REGRESSION and (loop.HISTORY / "turn_2.py").read_text() == WITH_YESTERDAY)
check("workspace: tampering with practice.py there changes nothing in lab/",
      (loop.LAB / "practice.py").read_text() == (ROOT / "lab" / "practice.py").read_text())

# Turn 3: Claude Code returned no structured reply, but model.py changed.
row = play(WITH_YESTERDAY, {"result": "I'd rather not say."})
check("invalid reply: no structured_output is 'invalid reply'", row["verdict"] == "invalid reply", row["verdict"])
check("invalid reply: practice is still measured, so the idea's result isn't lost",
      cells(row, *PRACTICE) == ["3.036", "2.88", "99.9"], cells(row, *PRACTICE))
check("invalid reply: it never reached the judge", cells(row, *VALIDATION) == ["-", "-", "-"], cells(row, *VALIDATION))
check("invalid reply: the bar doesn't rise", "Ideas judged: 1. The next must win 97.5%" in agent_ledger_top(), agent_ledger_top())
saved = json.loads((loop.HISTORY / "turn_3.json").read_text())
check("invalid reply: the raw reply is saved in history/", saved.get("agent_reply") == "I'd rather not say.", saved)

# Turns 4-8: replies the schema refuses. Each would otherwise be judged: model.py changed and submit is true.
good = reply(None, submit=True)
bad_replies = {
    "a null structured_output": None,
    "a missing field": {name: value for name, value in good.items() if name != "lesson"},
    "an extra field": {**good, "family": "temperature"},
    "submit as text": {**good, "submit": "yes"},
    "practice wins above 100": reply((3.036, 2.88, 250.0), submit=True),
}
for name, bad in bad_replies.items():
    row = play(WITH_YESTERDAY, {"structured_output": bad})
    check(f"invalid reply: {name}", cells(row, "verdict", *VALIDATION) == ["invalid reply", "-", "-", "-"], row)
check("invalid reply: five more and the bar still hasn't risen",
      "Ideas judged: 1. The next must win 97.5%" in agent_ledger_top(), agent_ledger_top())

# Turn 9: a valid reply that asks to submit, but model.py is the champion.
row = play(None, {"structured_output": reply(None, submit=True, idea="tabs\tand\nnew lines")})
check("no change: submit true with an unchanged model.py is 'no change'", row["verdict"] == "no change", row["verdict"])
check("no change: nothing is measured or judged", cells(row, *PRACTICE, *VALIDATION) == ["-"] * 6, row)
check("no change: reply_check says the agent didn't run practice.py", row["reply_check"] == "not run", row["reply_check"])
check("ledger: an idea with tabs and new lines stays one cell", row["idea"] == "tabs and new lines", row["idea"])

# Turn 10: model.py crashes in the loop's practice run.
CRASHES = '''
def fit(history) -> None:
    raise ValueError("SECRET-ERROR-TEXT")


def predict(state, history, today) -> float:
    return 100.0
'''
row = play(CRASHES, {"structured_output": reply(None, submit=True)})
check("crashed: a model.py that fails the loop's practice run is 'crashed'",
      cells(row, "verdict", *PRACTICE, *VALIDATION) == ["crashed"] + ["-"] * 6, row)

# Turn 11: model.py gets through the practice year, then crashes on the judge's.
CRASHES_LATER = REGRESSION.replace("    X = _features([today", '''    if today["date"].year == 2025:
        raise ValueError("SECRET-ERROR-TEXT")
    X = _features([today''')
row = play(CRASHES_LATER, {"structured_output": reply(None, submit=True)})
check("crashed: a model.py that fails the judge's run is 'crashed', with its practice numbers",
      cells(row, "verdict", *PRACTICE, *VALIDATION) == ["crashed", "3.036", "3.036", "0.0", "-", "-", "-"], row)
check("crashed: the agent's copy shows only the word, never the error text",
      "crashed" in loop.AGENT_LEDGER.read_text() and "SECRET" not in loop.AGENT_LEDGER.read_text() + loop.LEDGER.read_text())
check("crashed: the bar doesn't rise", "Ideas judged: 1. The next must win 97.5%" in agent_ledger_top(), agent_ledger_top())

# Turn 12: the same small gain as turn 2, submitted this time. It is only the 2nd idea judged.
row = play(WITH_YESTERDAY, {"structured_output": reply(None, submit=True)})
check("thrown away: a submitted idea that misses the bar is 'thrown away'", row["verdict"] == "thrown away", row["verdict"])
check("thrown away: it faced the 2nd idea's bar, 97.5%: nothing since turn 1 counted as a try",
      cells(row, *VALIDATION) == ["2.893", "67.91", "97.5"], cells(row, *VALIDATION))
check("thrown away: reply_check is 'not run' when the reply's practice is null", row["reply_check"] == "not run", row["reply_check"])
check("thrown away: now the bar rises", "Ideas judged: 2. The next must win 98.33%" in agent_ledger_top(), agent_ledger_top())
check("thrown away: the champion stays", (loop.LAB / "model.py").read_text() == REGRESSION)

# Turn 13: no model.py comes back at all.
row = play(MISSING, {"structured_output": reply(None, submit=True)})
check("no change: a missing model.py is 'no change' too", cells(row, "verdict", *VALIDATION) == ["no change", "-", "-", "-"], row)


# --- After 13 turns: nothing from the validation year has reached the agent ----------

full, agents = loop.LEDGER.read_text(), loop.AGENT_LEDGER.read_text()
val_numbers = ["5.782", "2.915", "2.893", "67.91"]  # every validation score and win share the judge gave above
check("full ledger: holds the validation numbers", all(number in full for number in val_numbers))
check("agent's copy: holds none of them", not any(number in agents for number in val_numbers),
      [number for number in val_numbers if number in agents])
check("agent's copy: one row per turn, with the verdicts",
      [line.split("\t")[5] for line in agents.splitlines()[3:]] ==
      ["start", "KEPT", "dropped on practice"] + ["invalid reply"] * 6 + ["no change", "crashed", "crashed", "thrown away", "no change"],
      [line.split("\t")[5] for line in agents.splitlines()[3:]])
check("workspace: the ledger the agent reads is that copy, not the full ledger",
      seen["ledger"].splitlines()[2].split("\t") == loop.AGENT_COLUMNS and "67.91" not in seen["ledger"], seen["ledger"].splitlines()[2])


# --- The schema Claude Code gets is the schema the loop checks -------------------------

sent = json.loads(loop.AGENT[loop.AGENT.index("--json-schema") + 1])
check("the agent's command passes --json-schema the loop's own REPLY_SCHEMA", sent == loop.REPLY_SCHEMA)
check("the agent's command still asks for --output-format json, with the tool limits",
      loop.AGENT[loop.AGENT.index("--output-format") + 1] == "json" and "--allowedTools" in loop.AGENT)


shutil.rmtree(TMP)
print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
sys.exit(1 if failures else 0)
