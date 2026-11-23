"""Make a fake town's daily electricity use: a long pretend future, and a data store to start from.

    uv run scripts/make_data.py                                # the toy's data
    uv run scripts/make_data.py --start 2006-01 --filler 300   # a long, wide table (tests/test_size.py)

--start and --filler add earlier days and extra columns. They never change a value of the
days and columns the toy began with.

It rebuilds two folders from nothing (see store.py for the layout):
  data/future/  every month from --start to 2042-12. Nothing but scripts/ingest.py reads it.
  data/store/   the months up to 2025-12, copied from the future. The loop and the judge
                read these. scripts/ingest.py adds later months, which is how time passes.

We write the recipe ourselves, so we know the best score any model could get
(scripts/answer_key.py). The days 2023-01-01 to 2025-12-31 are the ones the ledger, history/
and the findings in CLAUDE.md describe, so they must never change, whatever the settings:
they come from their own random generator, in the same order of draws as always.
tests/test_store.py checks them.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import store  # imported after the path is set: store.py lives in the repo root, not here
from ingest import as_month, ingest

STORE_THROUGH = "2025-12"  # the store's newest month on a fresh checkout. The pretend future runs on to 2042-12


def town(days, rng, spell_before=None) -> tuple[pd.DataFrame, float]:
    """Make the town's forecast and load for a run of days.

    Parameters
    ----------
    days : pd.DatetimeIndex
        The days to make, in order.
    rng : np.random.Generator
        Where the random numbers come from. They are drawn in a fixed order: the spells,
        then the forecast's misses, then the load's noise.
    spell_before : float, optional
        How far above (+) or below (-) the season the day before `days` was, in C, so a
        warm or cold spell carries over. Left out, the town starts with no spell.

    Returns
    -------
    pd.DataFrame
        One row per day: date, temp_forecast (C), load (MWh).
    float
        The spell on the last day, to carry into whatever days come next.
    """
    n = len(days)

    # Temperature: Tokyo-like seasons (about 6 C in late Jan, 28 C in early Aug),
    # plus warm and cold spells that last a few days.
    season = 17 - 11 * np.cos(2 * np.pi * (days.dayofyear.to_numpy() - 25) / 365.25)
    spell = np.zeros(n)
    if spell_before is not None:
        spell[0] = 0.7 * spell_before + rng.normal(0, 1.8)
    for i in range(1, n):
        spell[i] = 0.7 * spell[i - 1] + rng.normal(0, 1.8)
    temp = season + spell
    temp_forecast = temp + rng.normal(0, 1.0, n)  # the day-ahead forecast misses by about 1 C

    # The secret recipe (MWh per day). Load follows the ACTUAL temperature,
    # but a model only ever gets the FORECAST, so even a perfect model can't hit zero.
    heating = 1.8 * np.maximum(0, 16 - temp)   # below 16 C, heaters come on
    cooling = 2.5 * np.maximum(0, temp - 22)   # above 22 C, air conditioners come on
    dow = days.dayofweek.to_numpy()
    weekend = np.select([dow == 5, dow == 6], [-8.0, -12.0], 0.0)  # Saturday, Sunday
    load = 100 + heating + cooling + weekend + rng.normal(0, 3, n)

    table = pd.DataFrame({"date": days, "temp_forecast": temp_forecast.round(1), "load": load.round(1)})
    return table, float(spell[-1])


def make(start, filler) -> pd.DataFrame:
    """Make every day from `start` to the end of the pretend future.

    Parameters
    ----------
    start : str
        The first month, written like "2023-01".
    filler : int
        How many extra input columns of random numbers to add (filler_001, ...). They mean
        nothing; they make the table as wide as a real one.

    Returns
    -------
    pd.DataFrame
        One row per day: date, temp_forecast, load, then the filler columns.
    """
    # The days the toy began with, 2023-01-01 to 2025-12-31, always come from the same
    # generator and are always drawn as one run of 1,096 days. Everything else has its own
    # generator, so no setting can change a single draw of them.
    first, spell = town(pd.date_range("2023-01-01", "2025-12-31", freq="D"), np.random.default_rng(7))
    later, _ = town(pd.date_range("2026-01-01", "2042-12-31", freq="D"), np.random.default_rng(8), spell)
    days = pd.concat([first, later], ignore_index=True)
    if start < "2023-01":  # earlier days too. (2023-01-01 still starts with no spell, as it always has.)
        earlier, _ = town(pd.date_range(f"{start}-01", "2022-12-31", freq="D"), np.random.default_rng(6))
        days = pd.concat([earlier, days], ignore_index=True)
    days = days[days.date >= f"{start}-01"].reset_index(drop=True)  # a --start after 2023-01 drops the days before it
    noise = np.random.default_rng(9).normal(size=(len(days), filler))
    names = [f"filler_{i:03d}" for i in range(1, filler + 1)]
    return pd.concat([days, pd.DataFrame(noise, columns=names)], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make the fake town's data: data/future/ and data/store/.")
    parser.add_argument("--start", type=as_month, default="2023-01", help="the first month (default 2023-01)")
    parser.add_argument("--filler", type=int, default=0, help="extra input columns of random numbers (default 0)")
    settings = parser.parse_args()
    if settings.start > STORE_THROUGH:
        parser.error(f"--start must be {STORE_THROUGH} or earlier, so the store has a first month")

    days = make(settings.start, settings.filler)
    for folder in (store.FUTURE, store.STORE):  # start again from nothing
        shutil.rmtree(folder, ignore_errors=True)
    for name, month_days in days.groupby(days.date.dt.strftime("%Y-%m")):
        store.write(name, month_days, store.FUTURE)
    ingest(STORE_THROUGH)

    future, have = store.months(store.FUTURE), store.months()
    print(f"Made {len(days)} days, {future[0]} to {future[-1]}, in {store.FUTURE}.")
    print(f"The store, {store.STORE}, holds {have[0]} to {have[-1]}. Later months arrive with scripts/ingest.py.")
