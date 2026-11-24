"""Scores two models on the practice period, the newest 12 months at /data.

It scores them the way the judge scores its own period: walk forward a month at a time, then
reshuffle the period's 52 weeks 10,000 times.

    python practice.py champion.py model.py

It prints both average misses (MWh; lower is better) and how often the second model wins.
After your turn the loop measures the same three numbers itself, and those go in the ledger.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = (
    True  # as in the judge's sandbox: a model gets its history without a copy being made
)

# The practice period is the newest months at /data, this many of them. The judge counts the
# same period back from the as-of date: this number is its PRACTICE_MONTHS. Change them together.
PRACTICE_MONTHS = 12
WEEKS = (
    52  # the period in 7-day blocks: 12 months are 365 or 366 days, so the last block has 8 or 9
)
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
    model writes to it, pandas copies what it writes to, so data itself never changes.
    """
    model = importlib.import_module(Path(model_file).stem)
    misses = []
    for start in pd.date_range(
        end=data.date.max(), periods=PRACTICE_MONTHS, freq="MS"
    ):  # the first day of each practice month
        month = data[(data.date >= start) & (data.date < start + pd.offsets.MonthBegin(1))]
        state = model.fit(data.iloc[: month.index[0]])  # every row before the month
        for i, day in zip(month.index, month.to_dict("records"), strict=True):
            today = {column: value for column, value in day.items() if column != "load"}
            guess = float(model.predict(state, data.iloc[:i], today))  # every row before today
            if not np.isfinite(guess):  # the judge counts this as a crash
                raise ValueError(
                    f"{model_file}: prediction for {day['date']:%Y-%m-%d} is not a number"
                )
            misses.append(abs(day["load"] - guess))
    return np.array(misses)


def win_share(old, new) -> float:
    """Re-deal the period's 52 weeks 10,000 times and return the share the new model wins.

    A re-deal draws 52 weeks, some twice and some not at all. The new model wins a re-deal
    when its average miss is the lower one.
    Whole weeks move together because neighbouring days share weather.
    This is the judge's own reshuffle, with the judge's seed.
    """
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
    print(
        f"champion_mae: {old.mean():.3f}   average miss of {champion_file} on the practice period, in MWh"
    )
    print(f"model_mae:    {new.mean():.3f}   average miss of {model_file}")
    print(
        f"wins_pct:     {100 * win_share(old, new):.2f}   % of {RESHUFFLES:,} reshuffles of the practice period's weeks that {model_file} wins"
    )
