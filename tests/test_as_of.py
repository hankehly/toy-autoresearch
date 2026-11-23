"""Proves that the periods follow the as-of date, and that an advance moves them. Run it from the repo root:

    uv run pytest tests/test_as_of.py

Everything here happens in temporary folders: the test file's own data folder and as-of file
(see data_folder and as_of_file in conftest.py) and a copy of lab/. So data/, as_of.txt,
lab/, history/ and the ledgers are never touched. The judge and the model sandbox are the
real ones. Only the agent is replaced: a stand-in that changes model.py and returns a reply,
as Claude Code would, so no login token is needed. About 1 minute.

The sections follow one store and one as-of date through time, step by step, as a person
would: make_data.py, a turn, six months arrive, an advance, a turn, one more month arrives,
another advance, a turn, the holdout, a reset. Each step is a fixture that asks for the step
before it and records what it saw, so a test reads its step's record and can run alone.
"""

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import judge
import loop
import store

ROOT = Path(__file__).resolve().parent.parent


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
    no_token = {
        name: value for name, value in os.environ.items() if name != "CLAUDE_CODE_OAUTH_TOKEN"
    }
    return subprocess.run(
        [sys.executable, *command], cwd=ROOT, env=no_token, capture_output=True, text=True
    )


def periods() -> dict:
    """Ask the judge for the periods, with `judge.py periods`.

    Returns
    -------
    dict
        "as_of", the as-of date, and for each period its first month, its last month and
        how many months it has. If the judge stopped: "stopped", the line it stopped with.
    """
    printed = run("judge/judge.py", "periods")
    if printed.returncode != 0:
        return {"stopped": printed.stderr.strip()}
    found = json.loads(printed.stdout)
    return {
        name: months if name == "as_of" else (months[0], months[-1], len(months))
        for name, months in found.items()
    }


def score(period, model_path) -> float | None:
    """Score a model on one period, with `judge.py score`.

    Parameters
    ----------
    period : str
        "practice", "val" or "holdout".
    model_path : Path
        The model to score.

    Returns
    -------
    float or None
        Its average miss in MWh, or None if the judge printed no score.
    """
    return json.loads(run("judge/judge.py", "score", period, str(model_path)).stdout or "{}").get(
        "mae"
    )


# Turns 0, 1 and 3 of the example run, as in tests/test_loop.py.
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

CRASHES = """
def fit(history) -> None:
    raise ValueError("this model can't be measured")


def predict(state, history, today) -> float:
    return 100.0
"""

# This model says which is its newest month, as its guess: month=2024-06 comes back as 202406.
NAMES_ITS_NEWEST_MONTH = """
import os


def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return float(max(os.listdir("/data")).removeprefix("month=").replace("-", ""))
"""


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> dict[str, Path]:
    """Save the test models that the judge scores from a file.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        pytest's own fixture, which makes temporary folders.

    Returns
    -------
    dict[str, Path]
        Where each model was saved, by name: "baseline", "regression" and "newest".
    """
    folder = tmp_path_factory.mktemp("models")
    paths = {}
    for name, code in [
        ("baseline", BASELINE),
        ("regression", REGRESSION),
        ("newest", NAMES_ITS_NEWEST_MONTH),
    ]:
        paths[name] = folder / f"{name}.py"
        paths[name].write_text(code)
    return paths


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
    "2025-12": {
        "as_of": "2025-12",
        "training": ("2023-01", "2023-06", 6),
        "practice": ("2023-07", "2024-06", 12),
        "val": ("2024-07", "2025-06", 12),
        "holdout": ("2025-07", "2025-12", 6),
    },
    "2026-06": {
        "as_of": "2026-06",
        "training": ("2023-01", "2023-12", 12),
        "practice": ("2024-01", "2024-12", 12),
        "val": ("2025-01", "2025-12", 12),
        "holdout": ("2026-01", "2026-06", 6),
    },
}


# --- No date is written into the code or the brief ----------------------------------------
# The periods come from the as-of date, so a date typed into these files would go stale at the
# next advance. A written-out date is a four-digit year, or two month names side by side.
# lab/ledger.tsv is not searched: it is a record, and its as_of column holds months.

MONTH = "(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
WRITTEN_DATE = re.compile(rf"\b(?:19|20)\d\d\b|\b{MONTH}\W{{0,3}}{MONTH}\b")


def test_no_written_out_date_in_loop_py_judge_lab_or_docker():
    searched = [ROOT / "loop.py"]
    for folder in ("judge", "lab", "docker"):
        # the code, the brief and the sandboxes' files. Nothing else: a stray file that isn't text would stop the search
        searched += [
            path
            for path in sorted((ROOT / folder).rglob("*"))
            if path.suffix in (".py", ".md", ".conf") or path.name == "Dockerfile"
        ]
    dates_found = []
    for path in searched:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if WRITTEN_DATE.search(line):
                dates_found.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:60]}")
    assert dates_found == []


def test_the_search_itself_finds_a_year_a_month_beside_a_year_and_two_months_side_by_side():
    assert all(
        WRITTEN_DATE.search(text)
        for text in ("through 2024-06", "Jul 2023", "Jan-Jun", "July - December")
    )


# --- As-of date 2025-12: the periods are the ones the code used to list, so no score changes ---


@pytest.fixture(scope="module")
def made() -> subprocess.CompletedProcess:
    """Run make_data.py, as a person does once after a fresh checkout.

    Returns
    -------
    subprocess.CompletedProcess
        The finished run. The store then holds 2023-01 to 2025-12.
    """
    return run("scripts/make_data.py")


@pytest.fixture(scope="module")
def at_2025_12(made, models) -> dict:
    """Ask the judge for the periods and the two models' scores, under the as-of date 2025-12.

    Parameters
    ----------
    made : subprocess.CompletedProcess
        The step before: the store holds 2023-01 to 2025-12.
    models : dict[str, Path]
        The saved test models.

    Returns
    -------
    dict
        "periods", what `judge.py periods` printed, and "scores": for "baseline" and
        "regression", the practice, validation and holdout scores.
    """
    return {
        "periods": periods(),
        "scores": {
            name: tuple(score(period, models[name]) for period in ("practice", "val", "holdout"))
            for name in ("baseline", "regression")
        },
    }


def test_make_data_py_fills_a_temporary_store(made):
    assert made.returncode == 0, made.stderr[-300:]


def test_as_of_2025_12_the_periods_are_the_first_column_of_the_issues_table(at_2025_12):
    assert at_2025_12["periods"] == PERIODS["2025-12"]


@pytest.mark.parametrize("name", ["baseline", "regression"])
def test_as_of_2025_12_the_model_scores_as_it_always_has_on_practice_validation_and_holdout(
    name, at_2025_12
):
    assert at_2025_12["scores"][name] == EXPECTED["2025-12"][name]


# --- The judge stops when the store can't fill the periods -------------------------------


@pytest.fixture(scope="module")
def judge_stopped(at_2025_12, models, as_of_file) -> dict:
    """Write two as-of dates the store can't serve, ask the judge each time, and put 2025-12 back.

    Parameters
    ----------
    at_2025_12 : dict
        The step before: the store ends at 2025-12.
    models : dict[str, Path]
        The saved test models.
    as_of_file : Path
        The test file's as-of file.

    Returns
    -------
    dict
        "store_too_short", the line the judge stopped with under 2026-06, and "score_run",
        its attempt to score then. "nothing_to_learn_from", the line it stopped with under
        2025-06: counting back from there, the practice period starts at the store's first month.
    """
    as_of_file.write_text("2026-06\n")
    seen = {
        "store_too_short": periods().get("stopped", ""),
        "score_run": run("judge/judge.py", "score", "val", str(models["baseline"])),
    }
    as_of_file.write_text("2025-06\n")
    seen["nothing_to_learn_from"] = periods().get("stopped", "")
    as_of_file.write_text("2025-12\n")
    return seen


def test_the_judge_stops_with_one_line_if_the_store_does_not_reach_the_as_of_date(judge_stopped):
    stopped = judge_stopped["store_too_short"]
    assert len(stopped.splitlines()) == 1
    assert "2026-06" in stopped and "2025-12" in stopped


def test_and_it_scores_nothing_then(judge_stopped):
    assert judge_stopped["score_run"].returncode != 0
    assert judge_stopped["score_run"].stdout == ""


def test_the_judge_stops_with_one_line_if_the_store_has_no_month_before_the_practice_period(
    judge_stopped,
):
    stopped = judge_stopped["nothing_to_learn_from"]
    assert len(stopped.splitlines()) == 1
    assert "before the practice period, 2023-01 to 2023-12" in stopped


# --- A temporary lab for the loop to work on, and a stand-in for the agent -----------------

PRACTICE = ("practice_champion", "practice", "practice_wins")
VALIDATION = ("val_mae", "win_pct", "bar_pct")


@pytest.fixture(scope="module")
def lab(tmp_path_factory) -> Iterator[None]:
    """Point the loop at a temporary lab, history and ledgers, with the baseline as model.py.

    The loop's folders are pointed at copies, so the real lab/, history/ and ledgers stay
    as they are. They point back when the test file is done.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        pytest's own fixture, which makes temporary folders.
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
        yield


def play(model_code) -> tuple[dict[str, str], str]:
    """Run the next turn. The stand-in agent writes model_code over model.py and asks for it
    to be judged.

    Parameters
    ----------
    model_code : str
        The model the stand-in agent leaves in its workspace.

    Returns
    -------
    tuple[dict[str, str], str]
        The turn's row in the full ledger, and the brief (program.md) the stand-in agent
        found in its workspace.
    """
    found = {}

    def stand_in(work) -> dict:
        found["brief"] = (work / "program.md").read_text()
        (work / "model.py").write_text(model_code)
        return {
            "structured_output": {
                "idea": "an idea worth trying",
                "lesson": "What this turn taught me.",
                "practice": None,
                "submit": True,
            }
        }

    with pytest.MonkeyPatch.context() as patch:  # the real run_agent is back when the turn is done
        patch.setattr(loop, "run_agent", stand_in)
        loop.turn(loop.last_turn() + 1)
    return loop.read_ledger()[-1], found["brief"]


def cells(row, *names) -> list[str]:
    """Pick cells from a ledger row.

    Parameters
    ----------
    row : dict[str, str]
        A row of the full ledger.
    *names : str
        The columns wanted.

    Returns
    -------
    list[str]
        Their cells, in the order asked for.
    """
    return [row[name] for name in names]


def stops(action) -> str:
    """Run something that may stop the program, such as loop.advance.

    Parameters
    ----------
    action : Callable
        What to run.

    Returns
    -------
    str
        The line it stopped with, or "" if it ran to the end.
    """
    try:
        action()
    except SystemExit as stop:
        return str(stop.code)
    return ""


def agent_ledger() -> list[str]:
    """Read the agent's copy of the ledger.

    Returns
    -------
    list[str]
        Its lines: three comment lines, the column names, then one line per row.
    """
    return loop.AGENT_LEDGER.read_text().splitlines()


def in_agent_sandbox(work, *command) -> str:
    """Run a command in an agent sandbox with the agent's months at /data and no network.
    Its workspace holds practice.py, the regression as champion.py and WITH_YESTERDAY as model.py.

    Parameters
    ----------
    work : Path
        An empty folder, which becomes the sandbox's workspace.
    *command : str
        The command to run in the workspace.

    Returns
    -------
    str
        What the command printed, or the end of its error.
    """
    # pytest makes its folders readable by you alone. On Linux the sandbox's user is not you
    # (Docker on a Mac lets it in anyway), so let others read this one.
    work.chmod(0o755)
    shutil.copy(ROOT / "lab" / "practice.py", work / "practice.py")
    (work / "champion.py").write_text(REGRESSION)
    (work / "model.py").write_text(WITH_YESTERDAY)
    ran = subprocess.run(
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
    return ran.stdout or ran.stderr[-300:]


# --- Before any advance: every row records the as-of date 2025-12 --------------------------


@pytest.fixture(scope="module")
def turn_1(judge_stopped, lab) -> dict:
    """Start the ledgers (loop.start()) and play turn 1: the regression, under 2025-12.

    Parameters
    ----------
    judge_stopped : dict
        The step before: the as-of date is 2025-12 again.
    lab : None
        The temporary lab.

    Returns
    -------
    dict
        "row", turn 1's row in the full ledger; "rows", the full ledger; "agent_ledger", the
        lines of the agent's copy; "brief", the program.md in the agent's workspace; and
        "template", lab/program.md itself.
    """
    loop.start()
    row, brief = play(REGRESSION)
    return {
        "row": row,
        "rows": loop.read_ledger(),
        "agent_ledger": agent_ledger(),
        "brief": brief,
        "template": (loop.LAB / "program.md").read_text(),
    }


def test_turn_1_under_2025_12_the_regression_is_kept_with_the_scores_it_always_had(turn_1):
    assert cells(turn_1["row"], "turn", "verdict", "practice", "val_mae") == [
        "1",
        "KEPT",
        "3.036",
        "2.915",
    ]


def test_both_rows_so_far_record_the_as_of_date_they_were_measured_under(turn_1):
    assert [row["as_of"] for row in turn_1["rows"]] == ["2025-12", "2025-12"]


def test_the_agents_copy_has_the_as_of_column_too(turn_1):
    assert turn_1["agent_ledger"][3].split("\t")[:2] == ["turn", "as_of"]
    assert turn_1["agent_ledger"][5].split("\t")[:2] == ["1", "2025-12"]


def test_the_brief_in_the_workspace_has_its_dates_filled_in(turn_1):
    # data 2023-01 to 2024-06, practice 2023-07 to 2024-06
    brief = turn_1["brief"]
    assert "`/data` holds 2023-01 to 2024-06" in brief
    assert "practice period, 2023-07 to 2024-06" in brief
    assert "/data/month=2023-01/part.parquet" in brief
    assert "over the 12 months" in brief
    assert [line for line in brief.splitlines() if "{" in line] == []  # no {name} is left unfilled


def test_lab_program_md_itself_is_a_template_and_names_no_date(turn_1):
    assert "{data_first} to {data_last}" in turn_1["template"]


# --- Months that arrive move nothing ------------------------------------------------------


@pytest.fixture(scope="module")
def six_months_arrived(turn_1) -> dict:
    """Run `ingest.py --through 2026-06`, and ask where the periods and the agent's months are.

    Parameters
    ----------
    turn_1 : dict
        The step before: one kept idea, under the as-of date 2025-12.

    Returns
    -------
    dict
        "run", the finished ingest; "newest", the store's newest month; "periods", what
        `judge.py periods` printed; and "agent_newest", the newest month the agent has.
    """
    ingest = run("scripts/ingest.py", "--through", "2026-06")
    return {
        "run": ingest,
        "newest": store.months()[-1],
        "periods": periods(),
        "agent_newest": loop.agent_months()[-1],
    }


def test_ingest_through_2026_06_adds_six_months_to_the_store(six_months_arrived):
    assert six_months_arrived["run"].returncode == 0, six_months_arrived["run"].stderr[-300:]
    assert six_months_arrived["newest"] == "2026-06"


def test_the_periods_stay_where_they_were_the_as_of_date_is_still_2025_12(six_months_arrived):
    assert six_months_arrived["periods"] == PERIODS["2025-12"]


def test_the_agents_months_still_end_at_2024_06(six_months_arrived):
    assert six_months_arrived["agent_newest"] == "2024-06"


# --- loop.advance(): an advance happens whole or not at all ------------------------------


def interrupted() -> None:
    """Stands in for loop.write_agent_ledger() when Ctrl-C arrives just after the full ledger got its row."""
    raise KeyboardInterrupt


def files_an_advance_writes(as_of_file) -> list[bytes]:
    """Read the three files an advance writes: the as-of file and the two ledgers.

    Parameters
    ----------
    as_of_file : Path
        The test file's as-of file.

    Returns
    -------
    list[bytes]
        The contents of the as-of file, the full ledger and the agent's copy, in that order.
    """
    return [path.read_bytes() for path in (as_of_file, loop.LEDGER, loop.AGENT_LEDGER)]


@pytest.fixture(scope="module")
def advances_cut_short(six_months_arrived, as_of_file) -> dict:
    """Try two advances that can't finish: one whose champion crashes, one interrupted while
    it writes the ledgers. The champion is the regression again afterwards.

    Parameters
    ----------
    six_months_arrived : dict
        The step before: the store reaches 2026-06, and the as-of date is still 2025-12.
    as_of_file : Path
        The test file's as-of file.

    Returns
    -------
    dict
        "before", the three files an advance writes, as they were. "crashed", the line the
        first advance stopped with, and "after_crashed", the three files after it.
        "after_interrupted", the three files after the second.
    """
    seen = {"before": files_an_advance_writes(as_of_file)}

    (loop.LAB / "model.py").write_text(CRASHES)
    seen["crashed"] = stops(loop.advance)
    seen["after_crashed"] = files_an_advance_writes(as_of_file)
    (loop.LAB / "model.py").write_text(REGRESSION)

    with pytest.MonkeyPatch.context() as patch, pytest.raises(KeyboardInterrupt):
        patch.setattr(loop, "write_agent_ledger", interrupted)
        loop.advance()
    seen["after_interrupted"] = files_an_advance_writes(as_of_file)
    return seen


def test_an_advance_that_cant_measure_the_champion_stops_and_leaves_the_as_of_date_and_the_ledger_as_they_were(
    advances_cut_short,
):
    assert "could not be measured" in advances_cut_short["crashed"]
    assert advances_cut_short["before"][0] == b"2025-12\n"
    assert advances_cut_short["after_crashed"] == advances_cut_short["before"]


def test_an_advance_interrupted_while_it_writes_the_ledgers_leaves_the_as_of_date_and_both_ledgers_as_they_were(
    advances_cut_short,
):
    assert advances_cut_short["after_interrupted"] == advances_cut_short["before"]


# --- loop.advance(): the as-of date moves to the store's newest month ---------------------


@pytest.fixture(scope="module")
def advanced_to_2026_06(advances_cut_short, as_of_file) -> dict:
    """Advance: loop.advance(), with the store reaching 2026-06.

    Parameters
    ----------
    advances_cut_short : dict
        The step before: the as-of date is still 2025-12, and the champion is the regression.
    as_of_file : Path
        The test file's as-of file.

    Returns
    -------
    dict
        "stopped", the line the advance stopped with ("" if it ran to the end); "as_of", the
        text of the as-of file; "periods", what `judge.py periods` printed; "rows", the full
        ledger; "agent_ledger", the lines of the agent's copy; and "champion", lab/model.py.
    """
    stopped = stops(loop.advance)
    return {
        "stopped": stopped,
        "as_of": as_of_file.read_text(),
        "periods": periods(),
        "rows": loop.read_ledger(),
        "agent_ledger": agent_ledger(),
        "champion": (loop.LAB / "model.py").read_text(),
    }


def test_advance_as_of_txt_reads_2026_06(advanced_to_2026_06):
    assert advanced_to_2026_06["stopped"] == ""
    assert advanced_to_2026_06["as_of"] == "2026-06\n"


def test_advance_the_periods_are_the_second_column_of_the_issues_table(advanced_to_2026_06):
    assert advanced_to_2026_06["periods"] == PERIODS["2026-06"]


def test_advance_one_advanced_row_with_no_turn_number_under_the_new_as_of_date(advanced_to_2026_06):
    rows = advanced_to_2026_06["rows"]
    assert cells(rows[-1], "turn", "as_of", "verdict") == ["-", "2026-06", "advanced"]
    assert len(rows) == 3


def test_advance_the_row_holds_the_champions_scores_on_the_new_practice_and_validation_periods(
    advanced_to_2026_06,
):
    # 2.970 and 2.768
    row = advanced_to_2026_06["rows"][-1]
    assert (float(row["practice"]), float(row["val_mae"])) == EXPECTED["2026-06"]["regression"][:2]


def test_advance_old_rows_are_not_measured_again(advanced_to_2026_06, turn_1):
    assert advanced_to_2026_06["rows"][:2] == turn_1["rows"]


def test_advance_the_champions_model_is_unchanged(advanced_to_2026_06):
    assert advanced_to_2026_06["champion"] == REGRESSION


def test_agents_copy_the_top_line_gives_the_champions_new_practice_score_and_an_advance_is_not_a_try(
    advanced_to_2026_06,
):
    assert advanced_to_2026_06["agent_ledger"][:2] == [
        "# Champion: turn 1, practice 2.97 MWh",
        "# Ideas judged: 1. The next must win 97.5% of the judge's reshuffles.",
    ]


def test_agents_copy_the_advanced_row_shows_the_new_practice_score_and_no_validation_number(
    advanced_to_2026_06,
):
    assert advanced_to_2026_06["agent_ledger"][-1].split("\t")[3:7] == [
        "-",
        "2.97",
        "-",
        "advanced",
    ]
    assert "2.768" not in "\n".join(advanced_to_2026_06["agent_ledger"])


# --- After the advance: the agent and the judge work on the new periods --------------------


@pytest.fixture(scope="module")
def turn_2(advanced_to_2026_06, as_of_file, tmp_path_factory) -> dict:
    """Under the as-of date 2026-06: look into the agent sandbox, play turn 2 (WITH_YESTERDAY),
    and try a second advance.

    Parameters
    ----------
    advanced_to_2026_06 : dict
        The step before: the as-of date is 2026-06.
    as_of_file : Path
        The test file's as-of file.
    tmp_path_factory : pytest.TempPathFactory
        pytest's own fixture, which makes temporary folders.

    Returns
    -------
    dict
        "months", the month folders the agent sandbox holds; "practice", the three numbers
        practice.py prints there for the regression against WITH_YESTERDAY; "row" and
        "brief", turn 2's row and brief; "second_advance", the line a second advance stopped
        with; and "as_of" and "rows", the as-of file and the full ledger after it.
    """
    months = sorted(in_agent_sandbox(tmp_path_factory.mktemp("agent"), "ls", "/data").split())
    printed = in_agent_sandbox(
        tmp_path_factory.mktemp("agent"), "python", "practice.py", "champion.py", "model.py"
    )
    lines = [line.split() for line in printed.splitlines() if ":" in line]
    row, brief = play(WITH_YESTERDAY)
    return {
        "months": months,
        "practice": {words[0].rstrip(":"): float(words[1]) for words in lines},
        "row": row,
        "brief": brief,
        "second_advance": stops(loop.advance),
        "as_of": as_of_file.read_text(),
        "rows": loop.read_ledger(),
    }


def test_the_agent_sandbox_holds_data_through_2024_12_24_months_from_2023_01(turn_2):
    months = turn_2["months"]
    assert (len(months), months[:1], months[-1:]) == (24, ["month=2023-01"], ["month=2024-12"])


def test_practice_py_scores_the_newest_12_months_of_that_data(turn_2):
    # 2024-01 to 2024-12: 2.970 against 2.922, won 86.18%
    assert turn_2["practice"] == {"champion_mae": 2.97, "model_mae": 2.922, "wins_pct": 86.18}


def test_a_turn_after_the_advance_its_row_is_turn_2_under_the_as_of_date_2026_06(turn_2):
    assert cells(turn_2["row"], "turn", "as_of") == ["2", "2026-06"]


def test_that_turn_the_loop_measured_the_same_practice_numbers_as_practice_py_too_few_wins_to_be_judged(
    turn_2,
):
    assert cells(turn_2["row"], *PRACTICE, "verdict", *VALIDATION) == [
        "2.97",
        "2.922",
        "86.18",
        "dropped on practice",
        "-",
        "-",
        "-",
    ]


def test_that_turn_the_brief_named_the_new_dates(turn_2):
    # data 2023-01 to 2024-12 and practice 2024-01 to 2024-12
    assert "`/data` holds 2023-01 to 2024-12" in turn_2["brief"]
    assert "practice period, 2024-01 to 2024-12" in turn_2["brief"]


def test_a_second_advance_stops_nothing_newer_to_advance_to(turn_2):
    assert "Nothing to advance to" in turn_2["second_advance"]
    assert turn_2["as_of"] == "2026-06\n"
    assert len(turn_2["rows"]) == 4


# --- One more month arrives: before the next advance it is mounted nowhere -------------------


def newest_month_in_the_model_sandbox(model_path, period) -> float:
    """Start a model sandbox for one period and ask the model that names its newest month.

    Parameters
    ----------
    model_path : Path
        The saved NAMES_ITS_NEWEST_MONTH model.
    period : str
        The period the sandbox is opened for: "practice", "val" or "holdout".

    Returns
    -------
    float
        Its newest month folder, as a number: month=2024-06 is 202406.
    """
    sandbox = judge.open_model_sandbox(model_path, period)
    guess = judge.request(sandbox, {"op": "predict", "today": {"date": "2000-01-01"}})["guess"]
    sandbox.stdin.close()
    sandbox.wait()
    return guess


@pytest.fixture(scope="module")
def one_more_month_arrived(turn_2, models) -> dict:
    """Run `ingest.py --through 2026-07`, and ask what the sandboxes hold: the as-of date is still 2026-06.

    Parameters
    ----------
    turn_2 : dict
        The step before: the as-of date is 2026-06, and so is the store's newest month.
    models : dict[str, Path]
        The saved test models.

    Returns
    -------
    dict
        "newest", the store's newest month; "agent_newest", the newest month the agent has;
        and "model_sandbox_newest", the newest month in the model sandbox for each period.
    """
    run("scripts/ingest.py", "--through", "2026-07")
    return {
        "newest": store.months()[-1],
        "agent_newest": loop.agent_months()[-1],
        "model_sandbox_newest": {
            period: newest_month_in_the_model_sandbox(models["newest"], period)
            for period in ("practice", "val", "holdout")
        },
    }


def test_ingest_through_2026_07_adds_one_month(one_more_month_arrived):
    assert one_more_month_arrived["newest"] == "2026-07"


def test_the_agents_months_still_end_at_2024_12(one_more_month_arrived):
    assert one_more_month_arrived["agent_newest"] == "2024-12"


@pytest.mark.parametrize(
    "period, newest", [("practice", 202312), ("val", 202412), ("holdout", 202512)]
)
def test_model_sandbox_under_2026_06_its_newest_month_is_the_one_before_the_period(
    period, newest, one_more_month_arrived
):
    assert one_more_month_arrived["model_sandbox_newest"][period] == newest


# --- The next advance moves one month, and ideas are judged on the new validation period ----


@pytest.fixture(scope="module")
def advanced_to_2026_07(one_more_month_arrived, as_of_file) -> dict:
    """Advance again, with the store reaching 2026-07, and play turn 3 (WITH_YESTERDAY again).

    Parameters
    ----------
    one_more_month_arrived : dict
        The step before: the store reaches 2026-07, and the as-of date is 2026-06.
    as_of_file : Path
        The test file's as-of file.

    Returns
    -------
    dict
        "stopped", the line the advance stopped with ("" if it ran to the end); "as_of", the
        text of the as-of file; "advanced_row", the row the advance added; "agent_ledger_top",
        the first line of the agent's copy; and "row", turn 3's row.
    """
    stopped = stops(loop.advance)
    seen = {
        "stopped": stopped,
        "as_of": as_of_file.read_text(),
        "advanced_row": loop.read_ledger()[-1],
        "agent_ledger_top": agent_ledger()[0],
    }
    seen["row"], _ = play(WITH_YESTERDAY)
    return seen


def test_advance_after_one_more_month_as_of_txt_reads_2026_07(advanced_to_2026_07):
    assert advanced_to_2026_07["stopped"] == ""
    assert advanced_to_2026_07["as_of"] == "2026-07\n"


def test_its_row_holds_the_champions_scores_on_the_periods_one_month_on(advanced_to_2026_07):
    # 2.917 and 2.807
    row = advanced_to_2026_07["advanced_row"]
    assert cells(row, "as_of", "verdict") == ["2026-07", "advanced"]
    assert (float(row["practice"]), float(row["val_mae"])) == EXPECTED["2026-07"]["regression"][:2]


def test_agents_copy_the_top_line_follows_the_newest_advance(advanced_to_2026_07):
    assert advanced_to_2026_07["agent_ledger_top"] == "# Champion: turn 1, practice 2.917 MWh"


def test_a_turn_under_2026_07_the_same_idea_now_wins_95_17_pct_on_practice_so_it_is_judged(
    advanced_to_2026_07,
):
    # judged on the validation period 2025-02 to 2026-01
    assert cells(advanced_to_2026_07["row"], "turn", "as_of", *PRACTICE, "verdict") == [
        "3",
        "2026-07",
        "2.917",
        "2.846",
        "95.17",
        "thrown away",
    ]


def test_it_is_the_2nd_try_so_the_bar_is_97_5_pct(advanced_to_2026_07):
    # the try count is still the plain count of judged ideas
    assert cells(advanced_to_2026_07["row"], *VALIDATION) == ["2.812", "44.65", "97.5"]


# --- loop.holdout(): the holdout period of the as-of date, with validation measured now ------


@pytest.fixture(scope="module")
def holdout_printed(advanced_to_2026_07) -> str:
    """Run loop.holdout() under the as-of date 2026-07.

    Parameters
    ----------
    advanced_to_2026_07 : dict
        The step before: the as-of date is 2026-07, and one idea has been kept.

    Returns
    -------
    str
        What loop.holdout() printed.
    """
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        loop.holdout()
    return printed.getvalue()


def test_holdout_the_baseline_and_the_kept_model_each_with_its_validation_score_measured_now_and_its_holdout_score(
    holdout_printed,
):
    table = {
        line.split(":")[0]: line.split()[-2:]
        for line in holdout_printed.splitlines()
        if line.startswith("turn ")
    }
    assert table == {"turn 0": ["5.80", "5.88"], "turn 1": ["2.81", "2.72"]}


def test_holdout_it_says_which_months_those_are(holdout_printed):
    assert "validation is 2025-02 to 2026-01, holdout is 2026-02 to 2026-07" in holdout_printed


# --- loop.reset(): back to the as-of date in the start row; the store keeps its months -------


@pytest.fixture(scope="module")
def after_reset(holdout_printed, as_of_file) -> dict:
    """Reset: loop.reset().

    Parameters
    ----------
    holdout_printed : str
        The step before: the as-of date is 2026-07.
    as_of_file : Path
        The test file's as-of file.

    Returns
    -------
    dict
        "as_of", the text of the as-of file; "newest", the store's newest month; "model",
        lab/model.py; and "left_behind", which of the ledgers and history/ still exist.
    """
    loop.reset()
    return {
        "as_of": as_of_file.read_text(),
        "newest": store.months()[-1],
        "model": (loop.LAB / "model.py").read_text(),
        "left_behind": [
            path.name for path in (loop.LEDGER, loop.AGENT_LEDGER, loop.HISTORY) if path.exists()
        ],
    }


def test_reset_as_of_txt_reads_2025_12_again(after_reset):
    assert after_reset["as_of"] == "2025-12\n"


def test_reset_the_store_still_holds_the_ingested_months(after_reset):
    assert after_reset["newest"] == "2026-07"


def test_reset_the_baseline_is_back_and_the_ledgers_and_history_are_cleared(after_reset):
    assert after_reset["model"] == BASELINE
    assert after_reset["left_behind"] == []


# With no ledger yet, an advance starts one first, as `run` does, so the start row still
# records the as-of date the research started under. The baseline is measured before the
# ledger is made, so a baseline that can't be measured leaves nothing half made.


@pytest.fixture(scope="module")
def advanced_with_no_ledger(after_reset, as_of_file) -> dict:
    """With no ledger: try an advance whose baseline crashes, then advance, then reset again.

    Parameters
    ----------
    after_reset : dict
        The step before: no ledgers, the baseline as model.py, the as-of date 2025-12.
    as_of_file : Path
        The test file's as-of file.

    Returns
    -------
    dict
        "crashed", the line the first advance stopped with; "left_behind", which ledgers
        exist after it; and "as_of_after_crashed", the as-of file then. "stopped", the line
        the second advance stopped with ("" if it ran to the end), and "rows", the full
        ledger after it. "as_of_after_reset", the as-of file after the reset that follows.
    """
    (loop.LAB / "model.py").write_text(CRASHES)
    seen = {
        "crashed": stops(loop.advance),
        "left_behind": [path.name for path in (loop.LEDGER, loop.AGENT_LEDGER) if path.exists()],
        "as_of_after_crashed": as_of_file.read_text(),
    }
    (loop.LAB / "model.py").write_text(BASELINE)

    seen["stopped"] = stops(loop.advance)
    seen["rows"] = loop.read_ledger()
    loop.reset()
    seen["as_of_after_reset"] = as_of_file.read_text()
    return seen


def test_advance_with_no_ledger_and_a_baseline_that_cant_be_measured_it_stops_and_no_ledger_is_left_behind(
    advanced_with_no_ledger,
):
    assert "could not be measured" in advanced_with_no_ledger["crashed"]
    assert advanced_with_no_ledger["left_behind"] == []
    assert advanced_with_no_ledger["as_of_after_crashed"] == "2025-12\n"


def test_advance_with_no_ledger_the_baselines_start_row_is_measured_under_2025_12_then_the_advanced_row_under_2026_07(
    advanced_with_no_ledger,
):
    assert advanced_with_no_ledger["stopped"] == ""
    assert [
        cells(row, "turn", "as_of", "verdict", "practice", "val_mae")
        for row in advanced_with_no_ledger["rows"]
    ] == [
        ["0", "2025-12", "start", "5.656", "5.782"],
        ["-", "2026-07", "advanced", "5.65", "5.801"],
    ]


def test_so_a_reset_after_that_still_returns_to_2025_12(advanced_with_no_ledger):
    assert advanced_with_no_ledger["as_of_after_reset"] == "2025-12\n"
