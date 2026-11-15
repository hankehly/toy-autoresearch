"""Scores two models on the practice year, Jul 2023 - Jun 2024, the way the judge scores its
own year: walk forward a month at a time, then reshuffle the year's 52 weeks 10,000 times.

    python practice.py champion.py model.py

It prints both average misses (MWh; lower is better) and how often the second model wins.
After your turn the loop measures the same three numbers itself, and those go in the ledger.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True  # as in the judge's sandbox: a model gets its history without a copy being made

MONTHS = pd.date_range("2023-07-01", "2024-06-01", freq="MS")  # the practice year
WEEKS = 52           # the year in 7-day blocks (the last block has 9 days: 366 = 51 x 7 + 9)
RESHUFFLES = 10_000  # how many times we re-deal the weeks


def read_data() -> pd.DataFrame:
    """Read every month folder at /data as one table.

    Returns
    -------
    pd.DataFrame
        One row per day, oldest first, numbered from 0.
    """
    files = sorted(Path("/data").glob("month=*/part.parquet"))  # one file per month, oldest first
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def daily_misses(model_file, data) -> np.ndarray:
    """Walk forward one month at a time, exactly like the judge.
    fit() learns from every day before the month. predict() sees every day before today,
    plus every column of today except its load.
    The history a model gets is the rows of data before today, not a copy of them. If the
    model writes to it, pandas copies what it writes to, so data itself never changes."""
    model = importlib.import_module(Path(model_file).stem)
    misses = []
    for start in MONTHS:
        month = data[(data.date >= start) & (data.date < start + pd.offsets.MonthBegin(1))]
        state = model.fit(data.iloc[:month.index[0]])  # every row before the month
        for i, day in zip(month.index, month.to_dict("records")):
            today = {column: value for column, value in day.items() if column != "load"}
            guess = float(model.predict(state, data.iloc[:i], today))  # every row before today
            if not np.isfinite(guess):  # the judge counts this as a crash
                raise ValueError(f"{model_file}: prediction for {day['date']:%Y-%m-%d} is not a number")
            misses.append(abs(day["load"] - guess))
    return np.array(misses)


def win_share(old, new) -> float:
    """Re-deal the year's 52 weeks 10,000 times (some weeks twice, some not at all) and return
    the share of re-deals in which the new model's average miss is the lower one.
    Whole weeks move together because neighbouring days share weather.
    This is the judge's own reshuffle, with the judge's seed."""
    week = np.minimum(np.arange(len(old)) // 7, WEEKS - 1)
    gaps = np.bincount(week, weights=new - old, minlength=WEEKS)
    days = np.bincount(week, minlength=WEEKS)
    picks = np.random.default_rng(0).integers(0, WEEKS, size=(RESHUFFLES, WEEKS))
    return float((gaps[picks].sum(axis=1) / days[picks].sum(axis=1) < 0).mean())


if __name__ == "__main__":
    champion_file, model_file = sys.argv[1], sys.argv[2]
    data = read_data()
    old = daily_misses(champion_file, data)
    new = daily_misses(model_file, data)
    print(f"champion_mae: {old.mean():.3f}   average miss of {champion_file} on the practice year, in MWh")
    print(f"model_mae:    {new.mean():.3f}   average miss of {model_file}")
    print(f"wins_pct:     {100 * win_share(old, new):.2f}   % of {RESHUFFLES:,} reshuffles of the practice year's weeks that {model_file} wins")
