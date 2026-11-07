"""The frozen judge. The loop calls it; the agent never sees this folder.

    python judge/judge.py score dev   lab/model.py             # Jul 2024 - Jun 2025: what the agent is graded on
    python judge/judge.py score vault lab/model.py             # Jul - Dec 2025: opened once, at the end
    python judge/judge.py compare champion.py candidate.py 12  # keep or throw away the 12th idea tested (dev only)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = pd.read_csv(Path(__file__).parent / "full_data.csv", parse_dates=["date"])
MONTHS = {
    "dev": pd.date_range("2024-07-01", "2025-06-01", freq="MS"),
    "vault": pd.date_range("2025-07-01", "2025-12-01", freq="MS"),
}
WEEKS = 52           # the 365 scored days, in 7-day blocks (the last block has 8)
RESHUFFLES = 10_000  # how many times we re-deal the weeks
FLUKE_BUDGET = 0.05  # the k-th idea tested must win at least 1 - 5%/k of the re-deals


def load_model(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def daily_misses(model_path, split):
    """Walk forward one month at a time, like real life.
    fit() sees only days before the month. predict() sees only days before today,
    plus today's date and temperature forecast - never today's load."""
    model = load_model(model_path)
    misses = []
    for start in MONTHS[split]:
        state = model.fit(DATA[DATA.date < start].copy())
        month = DATA[(DATA.date >= start) & (DATA.date < start + pd.offsets.MonthBegin(1))]
        for day in month.itertuples():
            today = {"date": day.date, "temp_forecast": day.temp_forecast}
            pred = float(model.predict(state, DATA[DATA.date < day.date].copy(), today))
            if not np.isfinite(pred):
                raise ValueError(f"prediction for {day.date:%Y-%m-%d} is not a number")
            misses.append(abs(day.load - pred))
    return np.array(misses)


def compare(champion_path, candidate_path, k):
    """Re-deal the 52 scored weeks 10,000 times (some weeks twice, some not at all).
    Whole weeks move together because neighbouring days share weather.
    The k-th idea tested is kept only if its average miss is lower in at least 1 - 5%/k
    of the re-deals: 95% for the 1st, 99.5% for the 10th, 99.95% for the 100th.
    Over a whole night, that keeps the expected number of lucky keeps under one."""
    old = daily_misses(champion_path, "dev")
    new = daily_misses(candidate_path, "dev")
    week = np.minimum(np.arange(len(old)) // 7, WEEKS - 1)
    gaps = np.bincount(week, weights=new - old, minlength=WEEKS)
    days = np.bincount(week, minlength=WEEKS)
    picks = np.random.default_rng(0).integers(0, WEEKS, size=(RESHUFFLES, WEEKS))
    win_share = float((gaps[picks].sum(axis=1) / days[picks].sum(axis=1) < 0).mean())
    bar = 1 - FLUKE_BUDGET / k
    return {
        "champion_mae": round(float(old.mean()), 3),
        "candidate_mae": round(float(new.mean()), 3),
        "win_pct": round(100 * win_share, 2),
        "bar_pct": round(100 * bar, 2),
        "keep": win_share >= bar,
    }


if __name__ == "__main__":
    if sys.argv[1] == "score":
        split, path = sys.argv[2], sys.argv[3]
        print(json.dumps({"split": split, "mae": round(float(daily_misses(path, split).mean()), 3)}))
    elif sys.argv[1] == "compare":
        print(json.dumps(compare(sys.argv[2], sys.argv[3], int(sys.argv[4]))))
