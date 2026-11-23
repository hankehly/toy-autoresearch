"""Shows that the judge's path works on a long, wide table, and how long it takes. Run it
from the repo root, with -s so that the time is printed:

    uv run pytest tests/test_size.py -s

It builds a separate data store in the test file's own data folder (see data_folder in
conftest.py, so data/ is never touched): 20 years of days, 2006-01 to 2025-12, with 300
filler columns of random numbers. As one table that is 7,305 rows by 303 columns, about
18 MB. Then the real judge scores the baseline on the validation period in the real model
sandbox, which starts with the 18.5 years before that period. The as-of date is held at
2025-12 in a temporary file (see as_of_file in conftest.py).

This is not a gigabyte test. The toy has one row per day, so it cannot reach the size of
real data. It proves that a wide table gets through every step: the month folders, the
mounts, the runner and the messages. About 30 seconds.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import store

ROOT = Path(__file__).resolve().parent.parent


def run(*command) -> subprocess.CompletedProcess:
    """Run one of the repo's scripts from the repo root, the way a person would.

    Parameters
    ----------
    *command : str
        The script and its arguments, as typed after `uv run`.

    Returns
    -------
    subprocess.CompletedProcess
        The finished run, with what it printed.
    """
    return subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True)


# --- A separate store: 20 years, 300 filler columns -----------------------------------------


@pytest.fixture(scope="module")
def wide_store() -> subprocess.CompletedProcess:
    """Fill the test file's data folder with a long, wide store: make_data.py --start 2006-01 --filler 300.

    Returns
    -------
    subprocess.CompletedProcess
        The finished run.
    """
    return run("scripts/make_data.py", "--start", "2006-01", "--filler", "300")


def test_make_data_start_2006_01_filler_300_fills_a_store_with_20_years_of_months(wide_store):
    months = store.months()
    assert wide_store.returncode == 0, wide_store.stderr[-300:]
    assert (len(months), months[0], months[-1]) == (240, "2006-01", "2025-12")


def test_the_store_is_one_table_of_7305_days_by_303_columns_about_18_mb(wide_store):
    table = store.read(store.months())
    megabytes = table.memory_usage(index=False).sum() / 1e6
    assert table.shape == (7305, 303)
    assert 17 < megabytes < 19


# --- The judge scores the baseline on it, through the model sandbox -------------------------
# The baseline is the 7-day average. This one also checks what it is handed: a model that
# got a narrower table, a shorter history or fewer of today's columns would crash, not score.

BASELINE_THAT_CHECKS_WHAT_IT_GETS = """
import pandas as pd


def fit(history) -> None:
    assert history.shape[1] == 303, f"history has {history.shape[1]} columns"
    assert history["date"].iloc[0] == pd.Timestamp("2006-01-01"), f"history starts {history['date'].iloc[0]}"
    return None


def predict(state, history, today) -> float:
    assert len(today) == 302 and "filler_300" in today and "load" not in today, f"today has {len(today)} columns"
    assert history["date"].iloc[-1] == today["date"] - pd.Timedelta(days=1), "history doesn't end yesterday"
    return history["load"].tail(7).mean()
"""


def test_the_judge_scores_the_baseline_on_the_validation_period_with_every_column_reaching_the_model(
    wide_store, tmp_path
):
    model = tmp_path / "model.py"
    model.write_text(BASELINE_THAT_CHECKS_WHAT_IT_GETS)

    began = time.time()
    scored = run("judge/judge.py", "score", "val", str(model))
    seconds = time.time() - began

    assert scored.returncode == 0, scored.stderr[-300:]
    # 2023-01 to 2025-12 are the toy's own days whatever the settings, and the baseline reads only
    # the last 7 loads, so its score is the one in ledger.tsv: 5.782.
    assert json.loads(scored.stdout)["mae"] == 5.782
    print(
        f"\nTIME  scoring the validation period (one year) took {seconds:.1f} seconds: 365 predictions, each with a history of about 6,900 rows by 303 columns"
    )
