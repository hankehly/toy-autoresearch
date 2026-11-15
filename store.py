"""The data store: one folder per month, laid out the way Spark writes it.

    data/store/month=2023-01/part.parquet     columns: date, temp_forecast, load

The loop, the judge and the scripts read and write the store through these functions and
nothing else, so pointing the project at a real store later changes this file, plus the
three programs that run inside a sandbox and so can't import it: each reads the month
folders at /data itself (docker/model-sandbox/runner.py, lab/practice.py, lab/check.py).
A sandbox sees the month folders that mounts() gives it, read-only at /data, and no other month.

data/ is not in git. `uv run scripts/make_data.py` builds it.
"""
import os
import sys
from pathlib import Path

import pandas as pd

# TOY_DATA names a different data folder. The tests use it to work on a temporary store.
DATA = Path(os.environ.get("TOY_DATA") or Path(__file__).parent / "data").resolve()
STORE = DATA / "store"    # the months that have arrived: all the data the research can use
FUTURE = DATA / "future"  # the pretend future. Only scripts/ingest.py reads it


def months(folder=STORE) -> list[str]:
    """List the months a folder holds.

    Parameters
    ----------
    folder : Path
        The store, or the pretend future.

    Returns
    -------
    list[str]
        The months, oldest first, written like "2023-01". A month folder counts only once
        its file is finished, so a write that failed halfway leaves no month behind.
    """
    return sorted(path.parent.name.removeprefix("month=") for path in folder.glob("month=*/part.parquet"))


def read(months, folder=STORE) -> pd.DataFrame:
    """Read months as one table.

    Parameters
    ----------
    months : list[str]
        The months to read, written like "2023-01".
    folder : Path
        The store, or the pretend future.

    Returns
    -------
    pd.DataFrame
        One row per day, in the order the months were given.

    Raises
    ------
    FileNotFoundError
        If the folder doesn't hold one of the months.
    """
    tables = [pd.read_parquet(folder / f"month={month}" / "part.parquet") for month in months]
    return pd.concat(tables, ignore_index=True)


def write(month, days, folder=STORE) -> None:
    """Write one month's days as a month folder, replacing the folder's file if it exists.

    The month appears whole or not at all: the days are written to a second file, which
    takes the real file's name only once it is finished. If the write fails, the month is
    as it was before: absent, or holding its old days.

    Parameters
    ----------
    month : str
        The month, written like "2023-01".
    days : pd.DataFrame
        The month's days, one row per day.
    folder : Path
        The store, or the pretend future.
    """
    path = folder / f"month={month}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    unfinished = path.with_name("part.parquet.unfinished")
    days.to_parquet(unfinished, index=False)
    unfinished.replace(path)  # one step, so no reader ever sees half a file


def mounts(months) -> list[str]:
    """The `docker run` arguments that show a sandbox these months of the store.

    Parameters
    ----------
    months : list[str]
        The months the sandbox may see. Each must be in the store.

    Returns
    -------
    list[str]
        One read-only mount per month folder, at /data/month=<month>. The sandbox sees no
        other month, because no other month folder is mounted.
    """
    arguments = []
    for month in months:
        arguments += ["--mount", f"type=bind,source={STORE / f'month={month}'},target=/data/month={month},readonly"]
    return arguments


def need() -> None:
    """Stop with one line if the store has no months yet.

    Raises
    ------
    SystemExit
        If the store is missing or empty. The line says how to build it.
    """
    if not months():
        sys.exit(f"No data store at {STORE}. Run: uv run scripts/make_data.py")
