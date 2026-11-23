"""Proves that the data store holds the days it should, and that new months arrive only
through ingest. Run it from the repo root:

    uv run pytest tests/test_store.py

Everything here happens in the test file's own data folder (see data_folder in conftest.py),
so data/ is never touched. The scores come from the real judge and the real model sandbox,
with the as-of date held at 2025-12 in a temporary file (see as_of_file in conftest.py).
About 30 seconds.

The sections change one store, step by step, as a person would: no store, make_data.py,
ingest.py, a write that fails, make_data.py with other settings. Each step is a fixture that
asks for the step before it, so a test run alone still starts from the right store. The
tests of a section read the store as that section's step left it, so they run in the order
they are written, which is pytest's order.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import store

ROOT = Path(__file__).resolve().parent.parent

BASELINE = '''
def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return history["load"].tail(7).mean()
'''

# The ledger, history/ and the findings in CLAUDE.md describe the 1,096 days that used to be
# judge/full_data.csv. This is that file's SHA-256 (see it with: git show d7be1ad:judge/full_data.csv | shasum -a 256).
OLD_CSV_SHA256 = "c8222e82de68e38b777e9c4a1bd5e30c83db93ee37d6c1b1886ff4e27939f506"

# The scores the judge gave before the store existed: turn 0 in ledger.tsv, and the answer key.
KNOWN_SCORES = {"baseline": 5.782, "answer key": 2.83}


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


def sha256_as_csv(days) -> str:
    """Hash days the way the old CSV file held them, to compare them with that file.

    Parameters
    ----------
    days : pd.DataFrame
        One row per day: date, temp_forecast, load.

    Returns
    -------
    str
        The SHA-256 of the days written as CSV text, as `shasum -a 256` prints it.
    """
    as_csv = days.assign(date=days.date.dt.strftime("%Y-%m-%d")).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(as_csv.encode()).hexdigest()


def val_scores(baseline) -> dict[str, float]:
    """Score the two models whose scores we know, with `judge.py score val`.

    Parameters
    ----------
    baseline : Path
        The baseline model's file.

    Returns
    -------
    dict[str, float]
        Each model's average miss on the validation period, in MWh, by the model's name.
    """
    return {name: json.loads(run("judge/judge.py", "score", "val", str(path)).stdout or "{}").get("mae")
            for name, path in [("baseline", baseline), ("answer key", ROOT / "scripts" / "answer_key.py")]}


def file_bytes(folder) -> dict[str, bytes]:
    """Read every file under a folder, to compare the folder before and after.

    Parameters
    ----------
    folder : Path
        The folder to read.

    Returns
    -------
    dict[str, bytes]
        Each file's contents, by its path inside the folder.
    """
    return {str(path.relative_to(folder)): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()}


@pytest.fixture(scope="module")
def baseline(tmp_path_factory) -> Path:
    """Save the baseline model as a file, for the judge to score.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        pytest's own fixture, which makes temporary folders.

    Returns
    -------
    Path
        The model's file.
    """
    path = tmp_path_factory.mktemp("models") / "baseline.py"
    path.write_text(BASELINE)
    return path


# --- No store yet: the judge and the loop say how to make one -------------------------------

@pytest.mark.parametrize("command", [["judge/judge.py", "score", "val", "scripts/answer_key.py"],
                                     ["loop.py", "run", "1"],
                                     ["loop.py", "holdout"],
                                     ["loop.py", "advance"]], ids=lambda command: " ".join(command[:2]))
def test_without_a_store_the_command_stops_with_one_line_that_says_to_run_make_data(command):
    stopped = run(*command)
    assert stopped.returncode != 0
    assert stopped.stdout == ""
    assert len(stopped.stderr.strip().splitlines()) == 1, stopped.stderr.strip()[-200:]
    assert "uv run scripts/make_data.py" in stopped.stderr


# --- make_data.py: a pretend future, and a store that starts where the CSV files ended ---

@pytest.fixture(scope="module")
def made() -> subprocess.CompletedProcess:
    """Step 1: run make_data.py with its usual settings.

    Returns
    -------
    subprocess.CompletedProcess
        The finished run.
    """
    return run("scripts/make_data.py")


@pytest.fixture(scope="module")
def original_days(made) -> pd.DataFrame:
    """The store's days as make_data.py first made them, read before any later step.

    Parameters
    ----------
    made : subprocess.CompletedProcess
        Step 1, which must have run.

    Returns
    -------
    pd.DataFrame
        One row per day, 2023-01-01 to 2025-12-31.
    """
    return store.read(store.months())


def test_make_data_runs(made):
    assert made.returncode == 0, made.stderr[-300:]


def test_the_pretend_future_is_2023_01_to_2042_12_one_folder_per_month(made):
    future = store.months(store.FUTURE)
    assert (len(future), future[0], future[-1]) == (240, "2023-01", "2042-12")


def test_the_store_starts_with_2023_01_to_2025_12(made):
    stored = store.months()
    assert (len(stored), stored[0], stored[-1]) == (36, "2023-01", "2025-12")


def test_a_month_folder_is_laid_out_the_way_spark_writes_it(made):
    assert (store.STORE / "month=2023-01" / "part.parquet").is_file()


def test_the_stores_1096_days_are_the_old_full_data_csv_value_for_value(original_days):
    assert len(original_days) == 1096
    assert sha256_as_csv(original_days) == OLD_CSV_SHA256, f"columns {list(original_days.columns)}"


def test_the_days_after_2025_are_whole_months_of_the_same_three_columns(made):
    january = store.read(["2026-01"], store.FUTURE)
    assert len(january) == 31
    assert list(january.columns) == ["date", "temp_forecast", "load"]
    assert january.date.dt.strftime("%Y-%m").eq("2026-01").all()
    assert january.notna().all().all()


def test_the_judge_scores_the_validation_period_from_the_store_as_it_did_from_the_csv_file(made, baseline):
    assert val_scores(baseline) == KNOWN_SCORES


# --- ingest.py: time passes a whole month at a time ---------------------------------------

@pytest.fixture(scope="module")
def ingested(original_days) -> dict:
    """Step 2: run ingest.py --through 2026-06, and keep the folders' files from before and after.

    Parameters
    ----------
    original_days : pd.DataFrame
        The days of step 1, which must have run and been read first.

    Returns
    -------
    dict
        "run": the finished run. "months": the store's months afterwards.
        "store_before", "future_before", "store_after": every file of that folder, by path.
    """
    store_before, future_before = file_bytes(store.STORE), file_bytes(store.FUTURE)
    finished = run("scripts/ingest.py", "--through", "2026-06")
    return {"run": finished, "months": store.months(), "store_before": store_before,
            "future_before": future_before, "store_after": file_bytes(store.STORE)}


def test_ingest_through_2026_06_adds_the_six_months_the_store_didnt_have(ingested):
    assert ingested["run"].returncode == 0, ingested["run"].stderr[-300:]
    assert (len(ingested["months"]), ingested["months"][-1]) == (42, "2026-06")


def test_ingest_an_added_month_is_the_futures_month_day_for_day(ingested):
    assert store.read(["2026-06"]).equals(store.read(["2026-06"], store.FUTURE))


def test_ingest_never_changes_a_month_the_store_already_had(ingested):
    assert all(ingested["store_after"][name] == content for name, content in ingested["store_before"].items())


def test_ingest_touches_nothing_else(ingested):
    # the future is as it was, and the data folder holds only the two folders
    assert file_bytes(store.FUTURE) == ingested["future_before"]
    assert sorted(path.name for path in store.DATA.iterdir()) == ["future", "store"]


def test_ingest_with_an_older_month_adds_nothing_and_removes_nothing(ingested):
    run("scripts/ingest.py", "--through", "2026-03")
    assert file_bytes(store.STORE) == ingested["store_after"]


def test_after_ingest_the_judges_scores_are_the_same(ingested, baseline):
    # months that arrive move no period until the as-of date advances
    assert val_scores(baseline) == KNOWN_SCORES


# --- A write that fails must not leave a month that looks finished ---------------------------

class DaysThatFailHalfway:
    """Stands in for a month's days when the disk fills up: half a file is written, then the write fails."""

    def to_parquet(self, path, index) -> None:
        """Write half a file where a table would write its Parquet file, then fail.

        Parameters
        ----------
        path : Path
            Where store.write() asked for the file.
        index : bool
            Unused. A table's to_parquet() takes it.

        Raises
        ------
        OSError
            Always, as a full disk would.
        """
        Path(path).write_bytes(b"half a file")
        raise OSError("No space left on device")


def write_that_fails(month) -> None:
    """Try to write a month to the store with a write that fails halfway.

    Parameters
    ----------
    month : str
        The month, written like "2026-07".
    """
    with pytest.raises(OSError):
        store.write(month, DaysThatFailHalfway())


def test_a_write_that_fails_for_a_new_month_adds_no_month(ingested):
    write_that_fails("2026-07")  # a month the store doesn't have yet
    assert store.months()[-1] == "2026-06"


def test_ingest_completes_such_a_month_the_next_time(ingested):
    write_that_fails("2026-07")
    run("scripts/ingest.py", "--through", "2026-07")
    assert store.months()[-1] == "2026-07"
    assert store.read(["2026-07"]).equals(store.read(["2026-07"], store.FUTURE))


def test_a_write_that_fails_over_a_month_the_store_has_leaves_that_month_as_it_was(ingested):
    december = store.read(["2025-12"])
    write_that_fails("2025-12")
    assert store.read(["2025-12"]).equals(december)


def test_after_that_failed_write_the_whole_folder_still_reads_as_one_table(ingested):
    every_day = store.read(store.months())
    write_that_fails("2025-12")
    whole_folder = pd.read_parquet(store.STORE)  # how lab/program.md tells the agent to read /data
    assert len(whole_folder) == len(every_day)


# --- make_data.py with other settings: 2023-01 to 2025-12 are still the same days -----------

@pytest.fixture(scope="module")
def made_earlier_and_wider(original_days) -> subprocess.CompletedProcess:
    """Step 3: run make_data.py again, with --start 2021-07 --filler 2. It rebuilds the store.

    Parameters
    ----------
    original_days : pd.DataFrame
        The days of step 1, which must have been read before the store is rebuilt.

    Returns
    -------
    subprocess.CompletedProcess
        The finished run.
    """
    return run("scripts/make_data.py", "--start", "2021-07", "--filler", "2")


def test_make_data_start_2021_07_filler_2_makes_a_store_that_starts_earlier_with_two_more_columns(made_earlier_and_wider):
    stored = store.months()
    assert made_earlier_and_wider.returncode == 0, made_earlier_and_wider.stderr[-300:]
    assert (stored[0], stored[-1]) == ("2021-07", "2025-12")
    assert list(store.read(["2021-07"]).columns) == ["date", "temp_forecast", "load", "filler_001", "filler_002"]


def test_with_those_settings_2023_01_to_2025_12_are_still_the_old_full_data_csv(made_earlier_and_wider):
    days = store.read([month for month in store.months() if month >= "2023-01"])[["date", "temp_forecast", "load"]]
    assert len(days) == 1096
    assert sha256_as_csv(days) == OLD_CSV_SHA256


def test_make_data_start_2024_01_makes_a_store_that_starts_later_with_the_same_days_from_there_on(original_days):
    made_later = run("scripts/make_data.py", "--start", "2024-01")
    stored = store.months()
    assert made_later.returncode == 0, made_later.stderr[-300:]
    assert stored[0] == "2024-01"
    assert store.read(stored).equals(original_days[original_days.date >= "2024-01-01"].reset_index(drop=True))
