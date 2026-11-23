"""Proves that a turn of the loop is logged the way the ledgers promise. Run it from the repo root:

    uv run pytest tests/test_loop.py

Every turn here is a real loop.turn() on a temporary copy of lab/, with the real judge and
the real model sandbox. Only the agent is replaced: a stand-in that changes model.py and
returns a reply, as Claude Code would. So no login token is needed, and nothing in lab/,
history/ or the ledgers is touched. The as-of date is held at 2025-12 in a temporary file
(see as_of_file in conftest.py), so the scores expected here stay right after the real one
has advanced. About 1 minute.

The 14 turns are played once, in order, by the `turns` fixture, which records what each
turn left behind. Each test then reads one turn's record, so any test can run alone.
"""

import json
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

import loop
import store

ROOT = Path(__file__).resolve().parent.parent

# every test here needs the toy's data in the store (conftest.py)
pytestmark = pytest.mark.usefixtures("toy_store")


# --- Three models whose scores we know ----------------------------------------------
# They are turns 0, 1 and 3 of the example run. The expected numbers below come from that
# run's ledger (validation) and from issue #1's table (practice).

BASELINE = """
def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return history["load"].tail(7).mean()
"""

REGRESSION = """
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
"""

WITH_YESTERDAY = """
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
"""

# practice.py and check.py hand a model its history without copying it, as the judge's sandbox
# does. A write must stay in the table it was made to, or this model would crash: it checks,
# on every call, that nothing written in an earlier call is still there.
BASELINE_THAT_WRITES_TO_ITS_HISTORY = """
def fit(history) -> None:
    assert history["load"].min() > 0, "loads written in an earlier call are still there"
    history["dow"] = history["date"].dt.dayofweek  # a new column
    history.loc[:, "load"] = 0.0                    # and new values over the old ones
    return None


def predict(state, history, today) -> float:
    assert "dow" not in history.columns, "a column added in an earlier call is still there"
    assert history["load"].min() > 0, "loads written in an earlier call are still there"
    guess = history["load"].tail(7).mean()
    history["dow"] = history["date"].dt.dayofweek
    history.loc[:, "load"] = 0.0
    return guess
"""


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> dict[str, Path]:
    """Save the four models above as files, for the judge to read.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        pytest's own fixture, which makes temporary folders.

    Returns
    -------
    dict[str, Path]
        Each model's file, by a short name: "baseline", "regression", "yesterday", "writes".
    """
    folder = tmp_path_factory.mktemp("models")
    codes = {
        "baseline": BASELINE,
        "regression": REGRESSION,
        "yesterday": WITH_YESTERDAY,
        "writes": BASELINE_THAT_WRITES_TO_ITS_HISTORY,
    }
    for name, code in codes.items():
        (folder / f"{name}.py").write_text(code)
    return {name: folder / f"{name}.py" for name in codes}


# --- The judge can score the practice year, in the model sandbox ---------------------


def test_the_judge_scores_the_practice_year_the_regression_misses_by_3_036(models):
    assert loop.judge("score", "practice", models["regression"]).get("mae") == 3.036


def test_the_judge_compares_two_models_on_the_practice_year_with_no_bar(models):
    practice = loop.judge("practice", models["baseline"], models["regression"])
    assert practice == {"champion_mae": 5.656, "candidate_mae": 3.036, "win_pct": 100.0}


def test_the_judge_names_the_bar_for_the_7th_idea():
    assert loop.judge("bar", 7) == {"bar_pct": 99.29}


def test_the_judges_validation_verdict_is_unchanged(models):
    # turn 3 of the example run
    still = loop.judge("compare", models["regression"], models["yesterday"], 3)
    assert still == {
        "champion_mae": 2.915,
        "candidate_mae": 2.893,
        "win_pct": 67.91,
        "bar_pct": 98.33,
        "keep": False,
    }


# --- lab/practice.py gives the agent the same practice numbers ----------------------
# It runs where the agent runs it: in the agent sandbox, with the agent's months at /data,
# here with no network at all.


def in_agent_sandbox(work, champion_code, model_code, *command) -> subprocess.CompletedProcess:
    """Run a command in an agent sandbox, as the agent would.

    Parameters
    ----------
    work : Path
        An empty folder, which becomes the sandbox's workspace.
    champion_code : str
        The code saved as champion.py in the workspace.
    model_code : str
        The code saved as model.py there.
    *command : str
        The command to run in the workspace, which also holds practice.py and check.py.

    Returns
    -------
    subprocess.CompletedProcess
        The finished run, with what it printed.
    """
    # pytest makes its folders readable by you alone. On Linux the sandbox's user is not you
    # (Docker on a Mac lets it in anyway), so let others read this one.
    work.chmod(0o755)
    for name in ("practice.py", "check.py"):
        shutil.copy(ROOT / "lab" / name, work / name)
    (work / "champion.py").write_text(champion_code)
    (work / "model.py").write_text(model_code)
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "-v",
            f"{work}:/work",
            *store.mounts(loop.agent_months()),
            "toy-agent-sandbox",
            *command,
        ],
        capture_output=True,
        text=True,
    )


def run_practice(work, champion_code, model_code) -> dict:
    """What practice.py prints for these two models, as {name: number}, or its error."""
    run = in_agent_sandbox(
        work, champion_code, model_code, "python", "practice.py", "champion.py", "model.py"
    )
    lines = [line.split() for line in run.stdout.splitlines() if ":" in line]
    return {words[0].rstrip(":"): float(words[1]) for words in lines} or {
        "error": run.stderr[-300:]
    }


def test_practice_py_prints_3_036_for_turn_1s_model_as_the_judge_does(tmp_path):
    printed = run_practice(tmp_path, BASELINE, REGRESSION)
    assert printed == {"champion_mae": 5.656, "model_mae": 3.036, "wins_pct": 100.0}


def test_practice_py_and_the_judge_agree_on_a_close_call(tmp_path, models):
    # turn 3: 3.036 -> 2.880
    printed = run_practice(tmp_path, REGRESSION, WITH_YESTERDAY)
    judged = loop.judge("practice", models["regression"], models["yesterday"])
    assert printed == {
        "champion_mae": 3.036,
        "model_mae": 2.88,
        "wins_pct": judged.get("win_pct"),
    }, judged


def test_practice_py_a_model_that_writes_to_its_history_scores_as_the_baseline_does(tmp_path):
    printed = run_practice(tmp_path, BASELINE, BASELINE_THAT_WRITES_TO_ITS_HISTORY)
    assert printed == {"champion_mae": 5.656, "model_mae": 5.656, "wins_pct": 0.0}


def test_the_judge_a_model_that_writes_to_its_history_scores_5_656_there_too(models):
    assert loop.judge("score", "practice", models["writes"]).get("mae") == 5.656


def test_check_py_runs_that_model_on_the_agents_last_two_months_and_its_writes_dont_last_there_either(
    tmp_path,
):
    # the agent's last two months are 2024-05 and 2024-06: 61 days
    checked = in_agent_sandbox(
        tmp_path, BASELINE, BASELINE_THAT_WRITES_TO_ITS_HISTORY, "python", "check.py"
    )
    assert checked.stdout.strip() == "OK: model.py ran on 61 days without crashing.", (
        checked.stderr[-300:]
    )


# --- A temporary lab for the loop to work on, and the turns played on it ----------------


@dataclass
class After:
    """What the loop had left behind when a turn was over.

    Attributes
    ----------
    row : dict[str, str]
        The turn's row in the full ledger.
    ledger : str
        The full ledger's text.
    agent_ledger : list[str]
        The lines of the agent's copy of the ledger.
    champion : str
        The code of lab/model.py.
    workspace_files : list[str]
        The names of the files the stand-in agent found in its workspace.
    workspace_ledger : str
        The text of the ledger.tsv it found there.
    """

    row: dict[str, str]
    ledger: str
    agent_ledger: list[str]
    champion: str
    workspace_files: list[str]
    workspace_ledger: str


def after(workspace=None) -> After:
    """Record what the loop has left behind so far.

    Parameters
    ----------
    workspace : dict, optional
        What the stand-in agent found in its workspace this turn: "files" and "ledger".

    Returns
    -------
    After
        The record.
    """
    workspace = workspace or {"files": [], "ledger": ""}
    return After(
        row=loop.read_ledger()[-1],
        ledger=loop.LEDGER.read_text(),
        agent_ledger=loop.AGENT_LEDGER.read_text().splitlines(),
        champion=(loop.LAB / "model.py").read_text(),
        workspace_files=workspace["files"],
        workspace_ledger=workspace["ledger"],
    )


@pytest.fixture(scope="module")
def started(toy_store, tmp_path_factory) -> Iterator[After]:
    """Point the loop at a temporary lab, history and ledgers, and start them: loop.start().

    The loop's folders are pointed at copies, so the real lab/, history/ and ledgers stay
    as they are. They point back when the test file is done.

    Parameters
    ----------
    toy_store : None
        The toy's data, which the judge reads.
    tmp_path_factory : pytest.TempPathFactory
        pytest's own fixture, which makes temporary folders.

    Yields
    ------
    After
        What loop.start() left behind: the baseline's row, as turn 0.
    """
    folder = tmp_path_factory.mktemp("loop")
    with pytest.MonkeyPatch.context() as patch:  # undoes its changes when the test file is done
        patch.setattr(loop, "LAB", folder / "lab")
        patch.setattr(loop, "HISTORY", folder / "history")
        patch.setattr(loop, "LEDGER", folder / "ledger.tsv")
        patch.setattr(loop, "AGENT_LEDGER", folder / "lab" / "ledger.tsv")
        loop.LAB.mkdir()
        for name in ("program.md", "check.py", "practice.py"):
            shutil.copy(ROOT / "lab" / name, loop.LAB / name)
        (loop.LAB / "model.py").write_text(BASELINE)
        loop.start()
        yield after()


def reply(
    practice, submit, idea="an idea worth trying", lesson="What this turn taught me."
) -> dict:
    """A structured reply, as the agent would fill it in. practice is the three numbers
    (champion_mae, model_mae, wins_pct) it says practice.py printed, or None."""
    names = ("champion_mae", "model_mae", "wins_pct")
    return {
        "idea": idea,
        "lesson": lesson,
        "submit": submit,
        "practice": dict(zip(names, practice, strict=True)) if practice else None,
    }


MISSING = "delete model.py"  # in TURNS: the stand-in agent leaves no model.py behind

# Claude Code gave up on the structured reply. Its response then has "errors" and a failure
# subtype in place of "result" and "structured_output".
GAVE_UP = {
    "type": "result",
    "subtype": "error_max_structured_output_retries",
    "is_error": True,
    "errors": ["Failed to provide valid structured output after 5 attempts"],
    "total_cost_usd": 0.2,
}

# Replies the schema refuses. Each would otherwise be judged: model.py changed and submit is true.
GOOD_REPLY = reply(None, submit=True)
BAD_REPLIES = {
    "a null structured_output": None,
    "a missing field": {name: value for name, value in GOOD_REPLY.items() if name != "lesson"},
    "an extra field": {**GOOD_REPLY, "family": "temperature"},
    "submit as text": {**GOOD_REPLY, "submit": "yes"},
    "practice wins above 100": reply((3.036, 2.88, 250.0), submit=True),
}

CRASHES = """
def fit(history) -> None:
    raise ValueError("SECRET-ERROR-TEXT")


def predict(state, history, today) -> float:
    return 100.0
"""

CRASHES_LATER = WITH_YESTERDAY.replace(
    "    X = _features([today",
    """    if today["date"].year == 2025:
        raise ValueError("SECRET-ERROR-TEXT")
    X = _features([today""",
)

# The turns, by number. Each is what the stand-in agent writes over model.py (None: it leaves
# model.py alone; MISSING: it deletes it) and what Claude Code returns.
TURNS = {
    # a clear gain, submitted with the right practice numbers
    1: (
        REGRESSION,
        {
            "result": "Done.",
            "total_cost_usd": 0.1,
            "structured_output": reply(
                (5.656, 3.036, 100.0),
                submit=True,
                idea="weekday plus heating/cooling regression",
                lesson="Weekdays and temperature explain most of the load.",
            ),
        },
    ),
    # a changed model.py with submit false, and a practice number the agent got wrong
    2: (WITH_YESTERDAY, {"structured_output": reply((3.036, 2.5, 100.0), submit=False)}),
    # Claude Code gave up on the structured reply, but model.py changed
    3: (WITH_YESTERDAY, GAVE_UP),
    # turns 4 to 8: the five replies the schema refuses, in BAD_REPLIES's order
    **{
        4 + i: (WITH_YESTERDAY, {"structured_output": bad})
        for i, bad in enumerate(BAD_REPLIES.values())
    },
    # a valid reply that asks to submit, but model.py is the champion
    9: (None, {"structured_output": reply(None, submit=True, idea="tabs\tand\nnew lines")}),
    # model.py crashes in the loop's practice run
    10: (CRASHES, {"structured_output": reply(None, submit=True)}),
    # model.py wins enough on the practice year to be judged, then crashes on the judge's
    11: (CRASHES_LATER, {"structured_output": reply(None, submit=True)}),
    # the same small gain as turn 2, submitted this time. It is only the 2nd idea judged
    12: (WITH_YESTERDAY, {"structured_output": reply(None, submit=True)}),
    # no model.py comes back at all
    13: (MISSING, {"structured_output": reply(None, submit=True)}),
    # submitted, with practice numbers that would pass, but the loop measures 0% of reshuffles
    14: (BASELINE, {"structured_output": reply((3.036, 2.9, 95.0), submit=True)}),
}


def stand_in_agent(model_code, response, workspace) -> Callable[[Path], dict]:
    """Make one turn's stand-in for loop.run_agent(): what the agent does in its workspace.

    Parameters
    ----------
    model_code : str or None
        What it writes over model.py. None: it leaves model.py alone. MISSING: it deletes it.
    response : dict
        What Claude Code returns.
    workspace : dict
        Filled in with what the stand-in found in its workspace: "files" and "ledger".

    Returns
    -------
    Callable[[Path], dict]
        The stand-in. The loop calls it with the workspace's folder, as it calls run_agent().
    """

    def stand_in(work) -> dict:
        workspace["files"] = sorted(path.name for path in work.iterdir())
        workspace["ledger"] = (work / "ledger.tsv").read_text()
        # tampering must change nothing
        (work / "practice.py").write_text("print('champion_mae: 9.999')")
        if model_code == MISSING:
            (work / "model.py").unlink()
        elif model_code is not None:
            (work / "model.py").write_text(model_code)
        return response

    return stand_in


@pytest.fixture(scope="module")
def turns(started) -> Iterator[dict[int, After]]:
    """Play the 14 turns of TURNS on the temporary lab, in order, with a stand-in for the agent.

    Parameters
    ----------
    started : After
        The temporary lab, with the baseline's row in its ledgers.

    Yields
    ------
    dict[int, After]
        What each turn left behind, by the turn's number.
    """
    played = {}
    with pytest.MonkeyPatch.context() as patch:  # undoes its change when the test file is done
        for number, (model_code, response) in TURNS.items():
            workspace = {}  # what the stand-in agent finds in its workspace this turn
            patch.setattr(loop, "run_agent", stand_in_agent(model_code, response, workspace))
            loop.turn(number)
            played[number] = after(workspace)
        yield played


def cells(row, *names) -> list[str]:
    """The cells of a ledger row with these column names, in this order."""
    return [row[name] for name in names]


PRACTICE = ("practice_champion", "practice", "practice_wins")
VALIDATION = ("val_mae", "win_pct", "bar_pct")
# The second comment line of the agent's copy says how many ideas were judged, and the next bar.
ONE_JUDGED = "Ideas judged: 1. The next must win 97.5%"
TWO_JUDGED = "Ideas judged: 2. The next must win 98.33%"


# --- The two ledgers, after loop.start() --------------------------------------------------


def test_start_the_full_ledger_has_the_columns_issue_1_lists_and_issue_4s_as_of(started):
    assert list(started.row) == [
        "turn",
        "as_of",
        "idea",
        "practice_champion",
        "practice",
        "practice_wins",
        "val_mae",
        "win_pct",
        "bar_pct",
        "verdict",
        "lesson",
        "reply_check",
    ]


def test_start_the_baseline_row_has_its_as_of_date_its_practice_score_and_its_validation_score(
    started,
):
    assert cells(started.row, "turn", "as_of", "practice", "val_mae", "verdict") == [
        "0",
        "2025-12",
        "5.656",
        "5.782",
        "start",
    ]


def test_start_the_agents_copy_has_three_comment_lines(started):
    # they name the champion, the next bar and when to submit
    assert started.agent_ledger[:3] == [
        "# Champion: turn 0, practice 5.656 MWh",
        "# Ideas judged: 0. The next must win 95.0% of the judge's reshuffles.",
        f"# To be judged, an idea must win at least {loop.SUBMIT_PCT}% of the practice reshuffles.",
    ]


def test_start_the_agents_copy_has_no_validation_column(started):
    assert started.agent_ledger[3].split("\t") == [
        "turn",
        "as_of",
        "idea",
        "practice_champion",
        "practice",
        "practice_wins",
        "verdict",
        "lesson",
    ]


def test_start_the_agents_copy_has_the_baseline_row_without_its_validation_score(started):
    assert started.agent_ledger[4].split("\t") == [
        "0",
        "2025-12",
        "baseline: average of the last 7 days",
        "-",
        "5.656",
        "-",
        "start",
        "-",
    ]


# --- Turn 1: a clear gain, submitted with the right practice numbers ----------------------


def test_turn_1_a_submitted_idea_that_clears_the_bar_is_kept(turns):
    assert turns[1].row["verdict"] == "KEPT"


def test_turn_1_the_practice_numbers_are_the_loops_own(turns):
    assert cells(turns[1].row, *PRACTICE) == ["5.656", "3.036", "100.0"]


def test_turn_1_the_full_ledger_has_the_validation_numbers(turns):
    assert cells(turns[1].row, *VALIDATION) == ["2.915", "100.0", "95.0"]


def test_turn_1_the_idea_and_the_lesson_come_from_the_structured_reply(turns):
    assert cells(turns[1].row, "idea", "lesson") == [
        "weekday plus heating/cooling regression",
        "Weekdays and temperature explain most of the load.",
    ]


def test_turn_1_reply_check_is_ok_when_the_agents_practice_numbers_match_the_loops(turns):
    assert turns[1].row["reply_check"] == "ok"


def test_turn_1_model_py_is_the_new_champion(turns):
    assert turns[1].champion == REGRESSION


def test_turn_1_the_agents_copy_names_the_new_champion_and_a_higher_bar(turns):
    assert turns[1].agent_ledger[:2] == [
        "# Champion: turn 1, practice 3.036 MWh",
        "# Ideas judged: 1. The next must win 97.5% of the judge's reshuffles.",
    ]


def test_turn_1_the_agent_gets_practice_py_and_nothing_but_the_lab_files(turns):
    # the data is no longer copied into the workspace
    assert turns[1].workspace_files == [
        "check.py",
        "ledger.tsv",
        "model.py",
        "practice.py",
        "program.md",
    ]


def test_turn_1_history_saves_claude_codes_whole_response(turns):
    # the structured reply and the cost included
    saved = json.loads((loop.HISTORY / "turn_1.json").read_text())
    assert saved["response"]["structured_output"]["submit"] is True
    assert saved["response"]["total_cost_usd"] == 0.1


# --- Turn 2: a changed model.py with submit false, and a practice number the agent got wrong ---


def test_turn_2_a_changed_model_py_with_submit_false_is_dropped_on_practice(turns):
    assert turns[2].row["verdict"] == "dropped on practice"


def test_turn_2_its_practice_numbers_are_the_loops_not_the_agents_2_5(turns):
    assert cells(turns[2].row, *PRACTICE) == ["3.036", "2.88", "99.9"]


def test_turn_2_reply_check_flags_the_agents_wrong_number(turns):
    assert turns[2].row["reply_check"] == "practice mismatch"


def test_turn_2_it_never_reached_the_judge(turns):
    assert cells(turns[2].row, *VALIDATION) == ["-", "-", "-"]


def test_turn_2_the_bar_doesnt_rise(turns):
    assert ONE_JUDGED in turns[2].agent_ledger[1]


def test_turn_2_the_champion_stays_and_the_ideas_code_is_saved_in_history(turns):
    assert turns[2].champion == REGRESSION
    assert (loop.HISTORY / "turn_2.py").read_text() == WITH_YESTERDAY


def test_tampering_with_practice_py_in_the_workspace_changes_nothing_in_the_lab(turns):
    # the stand-in agent overwrites its practice.py on every turn
    assert (loop.LAB / "practice.py").read_text() == (ROOT / "lab" / "practice.py").read_text()


# --- Turn 3: Claude Code gave up on the structured reply, but model.py changed ------------


def test_turn_3_no_structured_output_is_an_invalid_reply(turns):
    assert turns[3].row["verdict"] == "invalid reply"


def test_turn_3_practice_is_still_measured_so_the_ideas_result_isnt_lost(turns):
    assert cells(turns[3].row, *PRACTICE) == ["3.036", "2.88", "99.9"]


def test_turn_3_it_never_reached_the_judge(turns):
    assert cells(turns[3].row, *VALIDATION) == ["-", "-", "-"]


def test_turn_3_the_bar_doesnt_rise(turns):
    assert ONE_JUDGED in turns[3].agent_ledger[1]


def test_turn_3_the_whole_response_is_saved_in_history_so_the_reason_isnt_lost(turns):
    saved = json.loads((loop.HISTORY / "turn_3.json").read_text())
    assert saved["response"] == GAVE_UP


# --- Turns 4 to 8: replies the schema refuses ---------------------------------------------


@pytest.mark.parametrize("number", [4, 5, 6, 7, 8], ids=list(BAD_REPLIES))
def test_turns_4_to_8_a_reply_the_schema_refuses_is_an_invalid_reply(number, turns):
    assert cells(turns[number].row, "verdict", *VALIDATION) == ["invalid reply", "-", "-", "-"]


def test_turns_4_to_8_five_more_invalid_replies_and_the_bar_still_hasnt_risen(turns):
    assert ONE_JUDGED in turns[8].agent_ledger[1]


# --- Turn 9: a valid reply that asks to submit, but model.py is the champion --------------


def test_turn_9_submit_true_with_an_unchanged_model_py_is_no_change(turns):
    assert turns[9].row["verdict"] == "no change"


def test_turn_9_nothing_is_measured_or_judged(turns):
    assert cells(turns[9].row, *PRACTICE, *VALIDATION) == ["-"] * 6


def test_turn_9_reply_check_says_the_agent_didnt_run_practice_py(turns):
    assert turns[9].row["reply_check"] == "not run"


def test_turn_9_an_idea_with_tabs_and_new_lines_stays_one_cell(turns):
    assert turns[9].row["idea"] == "tabs and new lines"


# --- Turn 10: model.py crashes in the loop's practice run ---------------------------------


def test_turn_10_a_model_py_that_fails_the_loops_practice_run_is_crashed(turns):
    assert cells(turns[10].row, "verdict", *PRACTICE, *VALIDATION) == ["crashed"] + ["-"] * 6


# --- Turn 11: model.py wins enough on the practice year to be judged, then crashes on the judge's ---


def test_turn_11_a_model_py_that_fails_the_judges_run_is_crashed_with_its_practice_numbers(turns):
    assert cells(turns[11].row, "verdict", *PRACTICE, *VALIDATION) == [
        "crashed",
        "3.036",
        "2.88",
        "99.9",
        "-",
        "-",
        "-",
    ]


def test_turn_11_the_ledgers_show_only_the_word_crashed_never_the_error_text(turns):
    agents_copy = "\n".join(turns[11].agent_ledger)
    assert "crashed" in agents_copy
    assert "SECRET" not in agents_copy + turns[11].ledger


def test_turn_11_the_bar_doesnt_rise(turns):
    assert ONE_JUDGED in turns[11].agent_ledger[1]


# --- Turn 12: the same small gain as turn 2, submitted this time ---------------------------


def test_turn_12_a_submitted_idea_that_misses_the_bar_is_thrown_away(turns):
    assert turns[12].row["verdict"] == "thrown away"


def test_turn_12_it_faced_the_2nd_ideas_bar_97_5(turns):
    # nothing since turn 1 counted as a try
    assert cells(turns[12].row, *VALIDATION) == ["2.893", "67.91", "97.5"]


def test_turn_12_reply_check_is_not_run_when_the_replys_practice_is_null(turns):
    assert turns[12].row["reply_check"] == "not run"


def test_turn_12_now_the_bar_rises(turns):
    assert TWO_JUDGED in turns[12].agent_ledger[1]


def test_turn_12_the_champion_stays(turns):
    assert turns[12].champion == REGRESSION


# --- Turn 13: no model.py comes back at all ------------------------------------------------


def test_turn_13_a_missing_model_py_is_no_change_too(turns):
    assert cells(turns[13].row, "verdict", *VALIDATION) == ["no change", "-", "-", "-"]


# --- Turn 14: submitted, with practice numbers that would pass, but the loop measures 0% ---


def test_turn_14_a_submitted_idea_that_wins_too_few_practice_reshuffles_is_dropped_on_practice(
    turns,
):
    # too few: less than loop.SUBMIT_PCT percent
    assert cells(turns[14].row, "verdict", *PRACTICE) == [
        "dropped on practice",
        "3.036",
        "5.656",
        "0.0",
    ]


def test_turn_14_the_agents_own_practice_numbers_cant_get_it_judged(turns):
    assert cells(turns[14].row, "reply_check", *VALIDATION) == ["practice mismatch", "-", "-", "-"]


def test_turn_14_the_bar_stays_where_turn_12_left_it(turns):
    assert TWO_JUDGED in turns[14].agent_ledger[1]


# --- After 14 turns: nothing from the validation period has reached the agent ----------

# every validation score and win share the judge gave above
VAL_NUMBERS = ["5.782", "2.915", "2.893", "67.91"]


def test_after_14_turns_the_full_ledger_holds_the_validation_numbers(turns):
    assert all(number in turns[14].ledger for number in VAL_NUMBERS)


def test_after_14_turns_the_agents_copy_holds_none_of_them(turns):
    agents_copy = "\n".join(turns[14].agent_ledger)
    assert [number for number in VAL_NUMBERS if number in agents_copy] == []


def test_after_14_turns_the_agents_copy_has_one_row_per_turn_with_the_verdicts(turns):
    verdicts = [line.split("\t")[6] for line in turns[14].agent_ledger[4:]]
    assert verdicts == (
        ["start", "KEPT", "dropped on practice"]
        + ["invalid reply"] * 6
        + ["no change", "crashed", "crashed", "thrown away", "no change", "dropped on practice"]
    )


def test_after_14_turns_the_ledger_the_agent_reads_is_that_copy_not_the_full_ledger(turns):
    assert turns[14].workspace_ledger.splitlines()[3].split("\t") == loop.AGENT_COLUMNS
    assert "67.91" not in turns[14].workspace_ledger


# --- The schema Claude Code gets is the schema the loop checks -------------------------


def test_the_agents_command_passes_json_schema_the_loops_own_reply_schema():
    sent = json.loads(loop.AGENT[loop.AGENT.index("--json-schema") + 1])
    assert sent == loop.REPLY_SCHEMA


def test_the_agents_command_still_asks_for_output_format_json_with_the_tool_limits():
    assert loop.AGENT[loop.AGENT.index("--output-format") + 1] == "json"
    assert "--allowedTools" in loop.AGENT
