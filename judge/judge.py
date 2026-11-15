"""The frozen judge. The loop calls it; the agent never sees this folder.

The agent's model.py never runs in here, next to the answers. It runs in the model sandbox:
a locked container (docker/model-sandbox) with no internet, and no files from your Mac except
model.py and the month folders before the first scored month, read-only. The judge sends it
messages, one line of JSON at a time, and reveals each day's actual load only after the
model has sent back its guess for that day.

    uv run judge/judge.py score val     lab/model.py           # validation, Jul 2024 - Jun 2025: what the agent is graded on
    uv run judge/judge.py score holdout lab/model.py           # holdout, Jul - Dec 2025: scored once, at the end
    uv run judge/judge.py compare champion.py candidate.py 12  # keep or throw away the 12th idea tested (validation only)
    uv run judge/judge.py practice champion.py candidate.py    # the same comparison on the practice year, with no bar
    uv run judge/judge.py bar 12                               # the share of reshuffles the 12th idea tested must win

The practice year (Jul 2023 - Jun 2024) is the agent's own: it has those months of the data
store and scores itself there with lab/practice.py. The loop asks the judge for the same
numbers, so the ledger records a measurement the agent can't touch.

The data comes from the data store (store.py). `uv run scripts/make_data.py` builds it.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import store  # noqa: E402  (imported after the path is set: store.py lives in the repo root, not here)

# TODO: Static date ranges. What to do as time passes and new data added?
MONTHS = {
    "practice": pd.date_range("2023-07-01", "2024-06-01", freq="MS"),  # the agent's own year: it has these answers
    "val": pd.date_range("2024-07-01", "2025-06-01", freq="MS"),      # validation: decides what's kept
    "holdout": pd.date_range("2025-07-01", "2025-12-01", freq="MS"),  # holdout: scored once, at the end
}
WEEKS = 52           # a scored year in 7-day blocks (the last block has 8 days; 9 in the practice year, which has a leap day)
RESHUFFLES = 10_000  # how many times we re-deal the weeks

# TODO: Where does this threshold come from? Was it used by a related project?
FLUKE_BUDGET = 0.05  # the k-th idea tested must win at least 1 - 5%/k of the re-deals


CRASHED = "model.py crashed or ran out of time (its error, if any, is printed above)"


def open_model_sandbox(model_path, split) -> subprocess.Popen:
    """Start a model sandbox, ready to be scored on one of the periods.

    Parameters
    ----------
    model_path : str or Path
        The model.py to run.
    split : str
        The period to be scored: "practice", "val" or "holdout". The sandbox gets the month
        folders before that period's first month, and the runner loads them as the model's
        history when it starts. No later month is in the sandbox.

    Returns
    -------
    subprocess.Popen
        The running sandbox. The judge talks to it through stdin and stdout.
    """
    first_scored = f"{MONTHS[split][0]:%Y-%m}"
    return subprocess.Popen([
        "docker", "run", "--rm", "-i", "--pull", "never",
        "--network", "none",                                          # no internet
        "--read-only", "--tmpfs", "/tmp",                             # can't change any file; /tmp vanishes afterwards
        "--user", "65534:65534",                                      # runs as "nobody", not as you
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",  # can't give itself extra powers
        "--cpus", "2", "--pids-limit", "64",                          # can't hog your Mac's processors (no memory cap: real data is big)
        "-v", f"{Path(model_path).resolve()}:/sandbox/model.py:ro",   # its one file of code, read-only
        *store.mounts([month for month in store.months() if month < first_scored]),  # its history, read-only at /data
        "toy-model-sandbox",
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def send(sandbox, message) -> None:
    """Send the sandbox a message; don't wait for a reply."""
    try:
        sandbox.stdin.write(json.dumps(message) + "\n")
        sandbox.stdin.flush()
    except BrokenPipeError:  # the sandbox has already closed
        raise RuntimeError(CRASHED) from None


def request(sandbox, message) -> dict:
    """Send the sandbox a message and return its reply."""
    send(sandbox, message)
    reply = sandbox.stdout.readline()
    if not reply:  # the sandbox closed without replying
        raise RuntimeError(CRASHED)
    return json.loads(reply)


def as_rows(days) -> list[dict]:
    """Days as plain rows that a message can carry: date, temp_forecast, load and any other column."""
    return days.assign(date=days.date.dt.strftime("%Y-%m-%d")).to_dict("records")


def daily_misses(model_path, split) -> np.ndarray:
    """Walk forward one month at a time, like real life.
    fit() learns from every day before the month. predict() sees every day before today,
    plus every column of today except its load. Each day's load is revealed only after the
    model has guessed it.
    The sandbox starts with the months before the first scored one already in it, so the
    judge reads only the scored months, one at a time."""
    sandbox = open_model_sandbox(model_path, split)
    misses = []
    for month in MONTHS[split]:
        request(sandbox, {"op": "fit"})
        for day in as_rows(store.read([f"{month:%Y-%m}"])):
            today = {column: value for column, value in day.items() if column != "load"}
            guess = float(request(sandbox, {"op": "predict", "today": today})["guess"])
            if not np.isfinite(guess):
                raise ValueError(f"prediction for {day['date']} is not a number")
            misses.append(abs(day["load"] - guess))
            send(sandbox, {"op": "reveal", "days": [day]})  # only now is today the past
    sandbox.stdin.close()  # no more messages: the runner finishes and Docker removes the container
    sandbox.wait()
    return np.array(misses)


def win_share(old, new) -> float:
    """Re-deal the year's 52 weeks 10,000 times (some weeks twice, some not at all) and return
    the share of re-deals in which the new model's average miss is the lower one.
    Whole weeks move together because neighbouring days share weather.
    lab/practice.py carries a copy of this function. Change them together."""
    week = np.minimum(np.arange(len(old)) // 7, WEEKS - 1)
    gaps = np.bincount(week, weights=new - old, minlength=WEEKS)
    days = np.bincount(week, minlength=WEEKS)
    picks = np.random.default_rng(0).integers(0, WEEKS, size=(RESHUFFLES, WEEKS))
    return float((gaps[picks].sum(axis=1) / days[picks].sum(axis=1) < 0).mean())


def bar(k) -> float:
    """The share of re-deals the k-th idea tested must win: 1 - 5%/k, so 95% for the 1st,
    99.5% for the 10th, 99.95% for the 100th.
    Over a whole night, that keeps the expected number of lucky keeps under one."""
    return 1 - FLUKE_BUDGET / k


def compare(champion_path, candidate_path, k) -> dict:
    """The verdict on the validation year. The k-th idea tested is kept only if its average
    miss is lower in at least bar(k) of the re-deals."""
    old = daily_misses(champion_path, "val")
    new = daily_misses(candidate_path, "val")
    share = win_share(old, new)
    return {
        "champion_mae": round(float(old.mean()), 3),
        "candidate_mae": round(float(new.mean()), 3),
        "win_pct": round(100 * share, 2),
        "bar_pct": round(100 * bar(k), 2),
        "keep": share >= bar(k),
    }


def practice(champion_path, candidate_path) -> dict:
    """The same comparison on the practice year, with no bar and no verdict. The loop logs
    these numbers, so the ledger never depends on the ones the agent reports."""
    old = daily_misses(champion_path, "practice")
    new = daily_misses(candidate_path, "practice")
    return {
        "champion_mae": round(float(old.mean()), 3),
        "candidate_mae": round(float(new.mean()), 3),
        "win_pct": round(100 * win_share(old, new), 2),
    }


if __name__ == "__main__":
    store.need()
    if sys.argv[1] == "score":
        split, path = sys.argv[2], sys.argv[3]
        print(json.dumps({"split": split, "mae": round(float(daily_misses(path, split).mean()), 3)}))
    elif sys.argv[1] == "compare":
        print(json.dumps(compare(sys.argv[2], sys.argv[3], int(sys.argv[4]))))
    elif sys.argv[1] == "practice":
        print(json.dumps(practice(sys.argv[2], sys.argv[3])))
    elif sys.argv[1] == "bar":
        print(json.dumps({"bar_pct": round(100 * bar(int(sys.argv[2])), 2)}))
