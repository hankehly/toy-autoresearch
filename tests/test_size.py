"""Shows that the judge's path works on a long, wide table, and how long it takes. Run it
from the repo root:

    uv run tests/test_size.py

It builds a separate data store in a temporary folder (TOY_DATA points the store there, so
data/ is never touched): 20 years of days, 2006-01 to 2025-12, with 300 filler columns of
random numbers. As one table that is 7,305 rows by 303 columns, about 18 MB. Then the real
judge scores the baseline on the validation period in the real model sandbox, which starts
with the 18.5 years before that period.

This is not a gigabyte test. The toy has one row per day, so it cannot reach the size of
real data. It proves that a wide table gets through every step: the month folders, the
mounts, the runner and the messages.
Prints PASS or FAIL for each check and the time the scoring took, and exits with 1 if any
check failed (about 30 seconds).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="test-size-"))
os.environ["TOY_DATA"] = str(TMP / "data")  # set before store is imported; the scripts and the judge started below follow it too
sys.path.insert(0, str(ROOT))
import loop  # noqa: E402  (imported after the path is set: these files live in the repo root, not here)
import store  # noqa: E402

loop.build()  # the judge runs the model in the model sandbox
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

made = run("scripts/make_data.py", "--start", "2006-01", "--filler", "300")
months = store.months()
check("make_data.py --start 2006-01 --filler 300 fills a store with 20 years of months",
      made.returncode == 0 and (len(months), months[0], months[-1]) == (240, "2006-01", "2025-12"),
      made.stderr[-300:] or (len(months), months[:1], months[-1:]))
table = store.read(months)
megabytes = table.memory_usage(index=False).sum() / 1e6
check("the store is one table of 7,305 days by 303 columns, about 18 MB",
      table.shape == (7305, 303) and 17 < megabytes < 19, f"{table.shape[0]:,} rows x {table.shape[1]} columns, {megabytes:.1f} MB")


# --- The judge scores the baseline on it, through the model sandbox -------------------------
# The baseline is the 7-day average. This one also checks what it is handed: a model that
# got a narrower table, a shorter history or fewer of today's columns would crash, not score.

BASELINE_THAT_CHECKS_WHAT_IT_GETS = '''
import pandas as pd


def fit(history) -> None:
    assert history.shape[1] == 303, f"history has {history.shape[1]} columns"
    assert history["date"].iloc[0] == pd.Timestamp("2006-01-01"), f"history starts {history['date'].iloc[0]}"
    return None


def predict(state, history, today) -> float:
    assert len(today) == 302 and "filler_300" in today and "load" not in today, f"today has {len(today)} columns"
    assert history["date"].iloc[-1] == today["date"] - pd.Timedelta(days=1), "history doesn't end yesterday"
    return history["load"].tail(7).mean()
'''
model = TMP / "model.py"
model.write_text(BASELINE_THAT_CHECKS_WHAT_IT_GETS)

began = time.time()
scored = run("judge/judge.py", "score", "val", str(model))
seconds = time.time() - began
mae = json.loads(scored.stdout or "{}").get("mae")
check("the judge scores the baseline on the validation period, with every column reaching the model",
      scored.returncode == 0 and isinstance(mae, float) and 3 < mae < 9, f"average miss {mae} MWh" if mae else scored.stderr[-300:])
print(f"TIME  scoring the validation period (one year) took {seconds:.1f} seconds: 365 predictions, each with a history of about 6,900 rows by 303 columns")


shutil.rmtree(TMP)
print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
sys.exit(1 if failures else 0)
