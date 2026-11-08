"""The frozen judge. The loop calls it; the agent never sees this folder.

The agent's model.py never runs in here, next to the answers. It runs in the model room:
a locked container (rooms/model-room) with no internet and no files except model.py.
The judge passes it notes, one line of JSON at a time, and reveals each day's actual
load only after the model has sent back its guess for that day.

    python judge/judge.py score val     lab/model.py           # validation, Jul 2024 - Jun 2025: what the agent is graded on
    python judge/judge.py score holdout lab/model.py           # holdout, Jul - Dec 2025: scored once, at the end
    python judge/judge.py compare champion.py candidate.py 12  # keep or throw away the 12th idea tested (validation only)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = pd.read_csv(Path(__file__).parent / "full_data.csv", parse_dates=["date"])
MONTHS = {
    "val": pd.date_range("2024-07-01", "2025-06-01", freq="MS"),      # validation: decides what's kept
    "holdout": pd.date_range("2025-07-01", "2025-12-01", freq="MS"),  # holdout: scored once, at the end
}
WEEKS = 52           # the 365 scored days, in 7-day blocks (the last block has 8)
RESHUFFLES = 10_000  # how many times we re-deal the weeks
FLUKE_BUDGET = 0.05  # the k-th idea tested must win at least 1 - 5%/k of the re-deals


CRASHED = "model.py crashed or ran out of time (its error, if any, is printed above)"


def open_model_room(model_path):
    """Start a model room for model_path. The judge talks to it through stdin and stdout."""
    return subprocess.Popen([
        "docker", "run", "--rm", "-i", "--pull", "never",
        "--network", "none",                                          # no internet
        "--read-only", "--tmpfs", "/tmp",                             # can't change any file; /tmp vanishes afterwards
        "--user", "65534:65534",                                      # runs as "nobody", not as you
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",  # can't give itself extra powers
        "--memory", "2g", "--cpus", "2", "--pids-limit", "64",        # can't hog your Mac
        "-v", f"{Path(model_path).resolve()}:/room/model.py:ro",      # its one file, read-only
        "toy-model-room",
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def tell(room, note):
    """Pass the room a note. No reply."""
    try:
        room.stdin.write(json.dumps(note) + "\n")
        room.stdin.flush()
    except BrokenPipeError:  # the room has already closed
        raise RuntimeError(CRASHED) from None


def ask(room, note):
    """Pass the room a note and wait for its reply."""
    tell(room, note)
    reply = room.stdout.readline()
    if not reply:  # the room closed without replying
        raise RuntimeError(CRASHED)
    return json.loads(reply)


def as_rows(days):
    """Days as plain rows that a note can carry: date, temp_forecast, load."""
    return days.assign(date=days.date.dt.strftime("%Y-%m-%d")).to_dict("records")


def daily_misses(model_path, split):
    """Walk forward one month at a time, like real life.
    fit() learns from every day before the month. predict() sees every day before today,
    plus today's date and temperature forecast - never today's load. Each day's load is
    revealed only after the model has guessed it."""
    room = open_model_room(model_path)
    tell(room, {"op": "reveal", "days": as_rows(DATA[DATA.date < MONTHS[split][0]])})
    misses = []
    for start in MONTHS[split]:
        ask(room, {"op": "fit"})
        month = DATA[(DATA.date >= start) & (DATA.date < start + pd.offsets.MonthBegin(1))]
        for day in month.itertuples():
            reply = ask(room, {"op": "predict", "date": f"{day.date:%Y-%m-%d}", "temp_forecast": day.temp_forecast})
            guess = float(reply["guess"])
            if not np.isfinite(guess):
                raise ValueError(f"prediction for {day.date:%Y-%m-%d} is not a number")
            misses.append(abs(day.load - guess))
            tell(room, {"op": "reveal", "days": as_rows(month[month.date == day.date])})  # only now is today the past
    room.stdin.close()  # no more notes: the runner finishes and Docker removes the container
    room.wait()
    return np.array(misses)


def compare(champion_path, candidate_path, k):
    """Re-deal the 52 scored weeks 10,000 times (some weeks twice, some not at all).
    Whole weeks move together because neighbouring days share weather.
    The k-th idea tested is kept only if its average miss is lower in at least 1 - 5%/k
    of the re-deals: 95% for the 1st, 99.5% for the 10th, 99.95% for the 100th.
    Over a whole night, that keeps the expected number of lucky keeps under one."""
    old = daily_misses(champion_path, "val")
    new = daily_misses(candidate_path, "val")
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
