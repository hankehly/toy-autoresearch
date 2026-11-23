"""Proves that the periods follow the as-of date, and that an advance moves them. Run it from the repo root:

    uv run tests/test_as_of.py

Everything here happens in a temporary folder: a data store (TOY_DATA points the store
there), an as-of file (TOY_AS_OF points the judge and the loop there) and a copy of lab/. So
data/, as_of.txt, lab/, history/ and the ledgers are never touched. The judge and the model
sandbox are the real ones. Only the agent is replaced: a stand-in that changes model.py and
returns a reply, as Claude Code would, so no login token is needed.
Prints PASS or FAIL for each check, and exits with 1 if any check failed (about 1 minute).
"""
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="test-as-of-"))
os.environ["TOY_DATA"] = str(TMP / "data")         # set before store and loop are imported; the scripts and the judge started below follow them too
os.environ["TOY_AS_OF"] = str(TMP / "as_of.txt")
sys.path[:0] = [str(ROOT), str(ROOT / "judge")]
import judge  # noqa: E402  (imported after the path is set: these files live in the repo, not here)
import loop  # noqa: E402
import store  # noqa: E402

AS_OF = TMP / "as_of.txt"
AS_OF.write_text("2025-12\n")  # what as_of.txt holds on a fresh checkout
loop.build()  # the judge runs every model in the model sandbox
failures = 0


def check(name, passed, detail="") -> None:
    """Print PASS or FAIL for one check, and count the failures.

    Parameters
    ----------
    name : str
        What the check proves.
    passed : bool
        Whether it held.
    detail : object, optional
        What was seen, printed after the name.
    """
    global failures
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def run(*command) -> subprocess.CompletedProcess:
    """Run one of the repo's scripts from the repo root, the way a person would, but with no
    login token: nothing here may start a real agent turn.

    Parameters
    ----------
    *command : str
        The script and its arguments, as typed after `uv run`.

    Returns
    -------
    subprocess.CompletedProcess
        The finished run, with what it printed.
    """
    no_token = {name: value for name, value in os.environ.items() if name != "CLAUDE_CODE_OAUTH_TOKEN"}
    return subprocess.run([sys.executable, *command], cwd=ROOT, env=no_token, capture_output=True, text=True)


def write_model(name, code) -> Path:
    """Save a test model as <TMP>/<name>.py and return its path."""
    path = TMP / f"{name}.py"
    path.write_text(code)
    return path


def periods() -> dict:
    """What `judge.py periods` prints: the as-of date, and each period's first month, last
    month and number of months. If the judge stopped, the line it stopped with."""
    printed = run("judge/judge.py", "periods")
    if printed.returncode != 0:
        return {"stopped": printed.stderr.strip()}
    found = json.loads(printed.stdout)
    return {name: months if name == "as_of" else (months[0], months[-1], len(months)) for name, months in found.items()}


def score(period, model_path) -> float | None:
    """A model's average miss on a period ("practice", "val" or "holdout"), as `judge.py score` prints it."""
    return json.loads(run("judge/judge.py", "score", period, str(model_path)).stdout or "{}").get("mae")


# Turns 0, 1 and 3 of the example run, as in tests/test_loop.py.
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

baseline, regression = write_model("baseline", BASELINE), write_model("regression", REGRESSION)

# The scores a model should get under each as-of date. Those for 2025-12 are the example run's
# (ledger.tsv and the README's table). The later ones were worked out once, by a separate
# script that had each period's months typed in by hand and never called the judge.
#                       practice  val    holdout
EXPECTED = {
    "2025-12": {"baseline": (5.656, 5.782, 5.819), "regression": (3.036, 2.915, 2.751)},
    "2026-06": {"baseline": (5.696, 5.802, 5.610), "regression": (2.970, 2.768, 2.670)},
    "2026-07": {"baseline": (5.650, 5.801, 5.877), "regression": (2.917, 2.807, 2.721)},
}

# The table in issue #4: each period's first month, last month and number of months.
PERIODS = {
    "2025-12": {"as_of": "2025-12", "training": ("2023-01", "2023-06", 6), "practice": ("2023-07", "2024-06", 12),
                "val": ("2024-07", "2025-06", 12), "holdout": ("2025-07", "2025-12", 6)},
    "2026-06": {"as_of": "2026-06", "training": ("2023-01", "2023-12", 12), "practice": ("2024-01", "2024-12", 12),
                "val": ("2025-01", "2025-12", 12), "holdout": ("2026-01", "2026-06", 6)},
}


# --- No date is written into the code or the brief ----------------------------------------
# The periods come from the as-of date, so a date typed into these files would go stale at the
# next advance. A written-out date is a four-digit year, or two month names side by side.
# lab/ledger.tsv is left out: it is a record, and its as_of column holds months.

MONTH = "(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
WRITTEN_DATE = re.compile(rf"\b(?:19|20)\d\d\b|\b{MONTH}\W{{0,3}}{MONTH}\b")
searched = [ROOT / "loop.py"] + [path for folder in ("judge", "lab", "docker") for path in sorted((ROOT / folder).rglob("*"))
                                 if path.is_file() and "__pycache__" not in path.parts and path != ROOT / "lab" / "ledger.tsv"]
found = [f"{path.relative_to(ROOT)}:{number}: {line.strip()[:60]}" for path in searched
         for number, line in enumerate(path.read_text().splitlines(), 1) if WRITTEN_DATE.search(line)]
check(f"no written-out date in loop.py, judge/, lab/ or docker/ ({len(searched)} files searched)", not found, found[:5])
check("the search itself finds a year, a month beside a year, and two months side by side",
      all(WRITTEN_DATE.search(text) for text in ("through 2024-06", "Jul 2023", "Jan-Jun", "July - December")))


# --- As-of date 2025-12: the periods are the ones the code used to list, so no score changes ---

made = run("scripts/make_data.py")
check("make_data.py fills a temporary store", made.returncode == 0, made.stderr[-300:])

check("as-of date 2025-12: the periods are the first column of the issue's table", periods() == PERIODS["2025-12"], periods())
for name, path in [("baseline", baseline), ("regression", regression)]:
    scores = (score("practice", path), score("val", path), score("holdout", path))
    check(f"as-of date 2025-12: the {name} scores as it always has on practice, validation and holdout",
          scores == EXPECTED["2025-12"][name], scores)


# --- The judge stops when the store can't fill the periods -------------------------------

AS_OF.write_text("2026-06\n")
stopped = periods().get("stopped", "")
check("the judge stops with one line if the store does not reach the as-of date",
      len(stopped.splitlines()) == 1 and "2026-06" in stopped and "2025-12" in stopped, stopped or periods())
stopped = run("judge/judge.py", "score", "val", str(baseline))
check("and it scores nothing then", stopped.returncode != 0 and stopped.stdout == "", stopped.stdout)

AS_OF.write_text("2025-06\n")  # counting back from here, the practice period starts at the store's first month
stopped = periods().get("stopped", "")
check("the judge stops with one line if the store has no month before the practice period",
      len(stopped.splitlines()) == 1 and "before the practice period, 2023-01 to 2023-12" in stopped, stopped or periods())
AS_OF.write_text("2025-12\n")


# --- A temporary lab for the loop to work on, and a stand-in for the agent -----------------
# The loop's folders are pointed at copies, so the real lab/, history/ and ledgers stay as they are.

loop.LAB, loop.HISTORY = TMP / "lab", TMP / "history"
loop.LEDGER, loop.AGENT_LEDGER = TMP / "ledger.tsv", TMP / "lab" / "ledger.tsv"
loop.LAB.mkdir()
for name in ("program.md", "check.py", "practice.py"):
    shutil.copy(ROOT / "lab" / name, loop.LAB / name)
(loop.LAB / "model.py").write_text(BASELINE)

seen = {}  # what the stand-in agent found in its workspace on the latest turn
PRACTICE = ("practice_champion", "practice", "practice_wins")
VALIDATION = ("val_mae", "win_pct", "bar_pct")


def play(model_code) -> dict[str, str]:
    """Run the next turn. The stand-in agent writes model_code over model.py and asks for it
    to be judged.

    Parameters
    ----------
    model_code : str
        The model the stand-in agent leaves in its workspace.

    Returns
    -------
    dict[str, str]
        The turn's row in the full ledger.
    """
    def stand_in(work) -> dict:
        seen["brief"] = (work / "program.md").read_text()
        (work / "model.py").write_text(model_code)
        return {"structured_output": {"idea": "an idea worth trying", "lesson": "What this turn taught me.",
                                      "practice": None, "submit": True}}

    loop.run_agent = stand_in
    loop.turn(loop.last_turn() + 1)
    return loop.read_ledger()[-1]


def cells(row, *names) -> list[str]:
    return [row[name] for name in names]


def stops(action) -> str:
    """Run something that should stop the program. Returns the line it stopped with, or "" if it didn't stop."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            action()
    except SystemExit as stop:
        return str(stop.code)
    return ""


def agent_ledger() -> list[str]:
    """The lines of the agent's copy of the ledger."""
    return loop.AGENT_LEDGER.read_text().splitlines()


def in_agent_sandbox(*command) -> str:
    """Run a command in an agent sandbox with the agent's months at /data and no network.
    Its workspace holds practice.py, the regression as champion.py and WITH_YESTERDAY as model.py.
    Returns what the command printed, or the end of its error."""
    work = Path(tempfile.mkdtemp(prefix="agent-", dir=TMP))
    shutil.copy(ROOT / "lab" / "practice.py", work / "practice.py")
    (work / "champion.py").write_text(REGRESSION)
    (work / "model.py").write_text(WITH_YESTERDAY)
    ran = subprocess.run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "-v", f"{work}:/work",
                          *store.mounts(loop.agent_months()), "toy-agent-sandbox", *command], capture_output=True, text=True)
    return ran.stdout or ran.stderr[-300:]


def practice_printed() -> dict[str, float]:
    """What practice.py prints in the agent sandbox for the regression against WITH_YESTERDAY, as {name: number}."""
    lines = [line.split() for line in in_agent_sandbox("python", "practice.py", "champion.py", "model.py").splitlines() if ":" in line]
    return {words[0].rstrip(":"): float(words[1]) for words in lines}


# --- Before any advance: every row records the as-of date 2025-12 --------------------------

loop.start()
row = play(REGRESSION)
check("turn 1 under 2025-12: the regression is KEPT, with the scores it always had",
      cells(row, "turn", "verdict", "practice", "val_mae") == ["1", "KEPT", "3.036", "2.915"], row)
check("both rows so far record the as-of date they were measured under, 2025-12",
      [r["as_of"] for r in loop.read_ledger()] == ["2025-12", "2025-12"], [r["as_of"] for r in loop.read_ledger()])
check("the agent's copy has the as_of column too",
      agent_ledger()[3].split("\t")[:2] == ["turn", "as_of"] and agent_ledger()[5].split("\t")[:2] == ["1", "2025-12"], agent_ledger()[3:])
check("the brief in the workspace has its dates filled in: data 2023-01 to 2024-06, practice 2023-07 to 2024-06",
      "`/data` holds 2023-01 to 2024-06" in seen["brief"] and "practice period, 2023-07 to 2024-06" in seen["brief"]
      and "/data/month=2023-01/part.parquet" in seen["brief"] and "over the 12 months" in seen["brief"]
      and "{" not in seen["brief"], [line for line in seen["brief"].splitlines() if "{" in line])
check("lab/program.md itself is a template and names no date", "{data_first} to {data_last}" in (loop.LAB / "program.md").read_text())


# --- Months that arrive move nothing ------------------------------------------------------

ingested = run("scripts/ingest.py", "--through", "2026-06")
check("ingest --through 2026-06 adds six months to the store", ingested.returncode == 0 and store.months()[-1] == "2026-06", ingested.stderr[-300:])
check("the periods stay where they were: the as-of date is still 2025-12", periods() == PERIODS["2025-12"], periods())
check("the agent's months still end at 2024-06", loop.agent_months()[-1] == "2024-06", loop.agent_months()[-1])


# --- loop.advance(): the as-of date moves to the store's newest month ---------------------

CRASHES = '''
def fit(history) -> None:
    raise ValueError("this model can't be measured")


def predict(state, history, today) -> float:
    return 100.0
'''
rows_before = loop.read_ledger()
(loop.LAB / "model.py").write_text(CRASHES)
stopped = stops(loop.advance)
check("an advance that can't measure the champion stops, and leaves the as-of date and the ledger as they were",
      "Not advanced" in stopped and AS_OF.read_text() == "2025-12\n" and loop.read_ledger() == rows_before, stopped or AS_OF.read_text())
(loop.LAB / "model.py").write_text(REGRESSION)

stopped = stops(loop.advance)
check("advance: as_of.txt reads 2026-06", stopped == "" and AS_OF.read_text() == "2026-06\n", stopped or AS_OF.read_text())
check("advance: the periods are the second column of the issue's table", periods() == PERIODS["2026-06"], periods())
row = loop.read_ledger()[-1]
check("advance: one `advanced` row, with no turn number, under the new as-of date",
      cells(row, "turn", "as_of", "verdict") == ["-", "2026-06", "advanced"] and len(loop.read_ledger()) == 3, row)
check("advance: the row holds the champion's scores on the new practice and validation periods, 2.970 and 2.768",
      (float(row["practice"]), float(row["val_mae"])) == EXPECTED["2026-06"]["regression"][:2], cells(row, "practice", "val_mae"))
check("advance: old rows are not measured again", loop.read_ledger()[:2] == rows_before)
check("advance: the champion's model is unchanged", (loop.LAB / "model.py").read_text() == REGRESSION)
check("agent's copy: the top line gives the champion's new practice score, and an advance is not a try",
      agent_ledger()[:2] == ["# Champion: turn 1, practice 2.97 MWh",
                             "# Ideas judged: 1. The next must win 97.5% of the judge's reshuffles."], agent_ledger()[:2])
check("agent's copy: the advanced row shows the new practice score and no validation number",
      agent_ledger()[-1].split("\t")[3:7] == ["-", "2.97", "-", "advanced"] and "2.768" not in loop.AGENT_LEDGER.read_text(), agent_ledger()[-1])


# --- After the advance: the agent and the judge work on the new periods --------------------

months = sorted(in_agent_sandbox("ls", "/data").split())
check("the agent sandbox holds data through 2024-12: 24 months from 2023-01",
      (len(months), months[:1], months[-1:]) == (24, ["month=2023-01"], ["month=2024-12"]), (len(months), months[:1], months[-1:]))
check("practice.py scores the newest 12 months of that data, 2024-01 to 2024-12: 2.970 against 2.922, won 86.18%",
      practice_printed() == {"champion_mae": 2.97, "model_mae": 2.922, "wins_pct": 86.18}, practice_printed())

row = play(WITH_YESTERDAY)
check("a turn after the advance: its row is turn 2, under the as-of date 2026-06",
      cells(row, "turn", "as_of") == ["2", "2026-06"], cells(row, "turn", "as_of"))
check("that turn: the loop measured the same practice numbers as practice.py, too few wins to be judged",
      cells(row, *PRACTICE, "verdict", *VALIDATION) == ["2.97", "2.922", "86.18", "dropped on practice", "-", "-", "-"], row)
check("that turn: the brief named the new dates, data 2023-01 to 2024-12 and practice 2024-01 to 2024-12",
      "`/data` holds 2023-01 to 2024-12" in seen["brief"] and "practice period, 2024-01 to 2024-12" in seen["brief"],
      [line for line in seen["brief"].splitlines() if "2024" in line])

stopped = stops(loop.advance)
check("a second advance stops: nothing newer to advance to",
      "Nothing to advance to" in stopped and AS_OF.read_text() == "2026-06\n" and len(loop.read_ledger()) == 4, stopped)


# --- One more month arrives: before the next advance it is mounted nowhere -------------------

run("scripts/ingest.py", "--through", "2026-07")
check("ingest --through 2026-07 adds one month", store.months()[-1] == "2026-07", store.months()[-1])
check("the agent's months still end at 2024-12", loop.agent_months()[-1] == "2024-12", loop.agent_months()[-1])

# This model says which is its newest month, as its guess: month=2024-06 comes back as 202406.
names_newest = write_model("newest", '''
import os


def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return float(max(os.listdir("/data")).removeprefix("month=").replace("-", ""))
''')
for period, newest in [("practice", 202312), ("val", 202412), ("holdout", 202512)]:
    sandbox = judge.open_model_sandbox(names_newest, period)
    guess = judge.request(sandbox, {"op": "predict", "today": {"date": "2000-01-01"}})["guess"]
    sandbox.stdin.close()
    sandbox.wait()
    check(f"model sandbox, scoring {period} under 2026-06: its newest month is the one before the period, {newest}",
          guess == newest, f"got {guess!r}")


# --- The next advance moves one month, and ideas are judged on the new validation period ----

stopped = stops(loop.advance)
row = loop.read_ledger()[-1]
check("advance after one more month: as_of.txt reads 2026-07", stopped == "" and AS_OF.read_text() == "2026-07\n", stopped or AS_OF.read_text())
check("its row holds the champion's scores on the periods one month on, 2.917 and 2.807",
      cells(row, "as_of", "verdict") == ["2026-07", "advanced"]
      and (float(row["practice"]), float(row["val_mae"])) == EXPECTED["2026-07"]["regression"][:2], row)
check("agent's copy: the top line follows the newest advance", agent_ledger()[0] == "# Champion: turn 1, practice 2.917 MWh", agent_ledger()[0])

row = play(WITH_YESTERDAY)
check("a turn under 2026-07: the same idea now wins 95.17% on practice, so it is judged, on 2025-02 to 2026-01",
      cells(row, "turn", "as_of", *PRACTICE, "verdict") == ["3", "2026-07", "2.917", "2.846", "95.17", "thrown away"], row)
check("it is the 2nd try: the try count is still the plain count of judged ideas, so the bar is 97.5%",
      cells(row, *VALIDATION) == ["2.812", "44.65", "97.5"], cells(row, *VALIDATION))


# --- loop.holdout(): the holdout period of the as-of date, with validation measured now ------

printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    loop.holdout()
table = {line.split(":")[0]: line.split()[-2:] for line in printed.getvalue().splitlines() if line.startswith("turn ")}
check("holdout: the baseline and the kept model, each with its validation score measured now and its holdout score",
      table == {"turn 0": ["5.80", "5.88"], "turn 1": ["2.81", "2.72"]}, printed.getvalue())
check("holdout: it says which months those are", "validation is 2025-02 to 2026-01, holdout is 2026-02 to 2026-07" in printed.getvalue(),
      printed.getvalue().splitlines()[:1])


# --- loop.reset(): back to the as-of date in the start row; the store keeps its months -------

loop.reset()
check("reset: as_of.txt reads 2025-12 again", AS_OF.read_text() == "2025-12\n", AS_OF.read_text())
check("reset: the store still holds the ingested months", store.months()[-1] == "2026-07", store.months()[-1])
check("reset: the baseline is back, and the ledgers and history are cleared",
      (loop.LAB / "model.py").read_text() == BASELINE and not loop.LEDGER.exists() and not loop.AGENT_LEDGER.exists() and not loop.HISTORY.exists())

# With no ledger yet, an advance starts one first, as `run` does, so the start row still
# records the as-of date the research started under.
stopped = stops(loop.advance)
rows = loop.read_ledger()
check("advance with no ledger: the baseline's start row is measured under 2025-12, then the advanced row under 2026-07",
      stopped == "" and [cells(r, "turn", "as_of", "verdict", "practice", "val_mae") for r in rows]
      == [["0", "2025-12", "start", "5.656", "5.782"], ["-", "2026-07", "advanced", "5.65", "5.801"]], stopped or rows)
loop.reset()
check("so a reset after that still returns to 2025-12", AS_OF.read_text() == "2025-12\n", AS_OF.read_text())


shutil.rmtree(TMP)
print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
sys.exit(1 if failures else 0)
