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

MONTHS = pd.date_range("2023-07-01", "2024-06-01", freq="MS")  # the practice year
WEEKS = 52           # the year in 7-day blocks (the last block has 9 days: 366 = 51 x 7 + 9)
RESHUFFLES = 10_000  # how many times we re-deal the weeks


def daily_misses(model_file, data) -> np.ndarray:
    """Walk forward one month at a time, exactly like the judge.
    fit() learns from every day before the month. predict() sees every day before today,
    plus today's date and temperature forecast - never today's load."""
    model = importlib.import_module(Path(model_file).stem)
    misses = []
    for start in MONTHS:
        state = model.fit(data[data.date < start].copy())
        month = data[(data.date >= start) & (data.date < start + pd.offsets.MonthBegin(1))]
        for day in month.itertuples():
            today = {"date": day.date, "temp_forecast": day.temp_forecast}
            guess = float(model.predict(state, data[data.date < day.date].copy(), today))
            if not np.isfinite(guess):  # the judge counts this as a crash
                raise ValueError(f"{model_file}: prediction for {day.date:%Y-%m-%d} is not a number")
            misses.append(abs(day.load - guess))
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
    data = pd.read_csv("data.csv", parse_dates=["date"])
    old = daily_misses(champion_file, data)
    new = daily_misses(model_file, data)
    print(f"champion_mae: {old.mean():.3f}   average miss of {champion_file} on the practice year, in MWh")
    print(f"model_mae:    {new.mean():.3f}   average miss of {model_file}")
    print(f"wins_pct:     {100 * win_share(old, new):.2f}   % of {RESHUFFLES:,} reshuffles of the practice year's weeks that {model_file} wins")
