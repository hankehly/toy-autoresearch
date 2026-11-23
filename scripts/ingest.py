"""Let time pass: move months from the pretend future into the data store.

    uv run scripts/ingest.py --through 2026-06

Copies every month up to 2026-06 that the store does not have yet from data/future/ to
data/store/. It adds whole months only, never changes a month the store already has, and
touches nothing else.

In a real project the job that builds the data (dbt) does this. This script exists only so
the toy can let time pass.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import store  # imported after the path is set: store.py lives in the repo root, not here


def as_month(text) -> str:
    """Read a month typed on the command line, so "2026-6" and "2026-06" mean the same.

    Parameters
    ----------
    text : str
        A month as typed.

    Returns
    -------
    str
        The month written like "2026-06".
    """
    return str(pd.Period(text, "M"))


def ingest(through) -> list[str]:
    """Copy the months that have arrived from the pretend future into the store.

    Parameters
    ----------
    through : str
        The newest month to copy, written like "2026-06".

    Returns
    -------
    list[str]
        The months added: those up to `through` that the store did not have.
    """
    have = store.months()
    added = [month for month in store.months(store.FUTURE) if month <= through and month not in have]
    for month in added:
        store.write(month, store.read([month], store.FUTURE))
    return added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move months from the pretend future into the data store.")
    parser.add_argument("--through", type=as_month, required=True, help="the newest month to add, like 2026-06")
    added = ingest(parser.parse_args().through)
    have = store.months()
    print(f"Added {len(added)} months" + (f" ({added[0]} to {added[-1]})" if added else "") + ". "
          + (f"The store holds {have[0]} to {have[-1]}." if have else "The store is empty: run scripts/make_data.py first."))
