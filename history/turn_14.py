"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, lo=17, hi=21) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(lo - t, 0), np.maximum(t - hi, 0)]
    return np.column_stack(cols)


def fit(history):
    """Least squares on weekday dummies plus heating/cooling degrees, thresholds by grid search."""
    y = history["load"].to_numpy(float)
    best = None
    for lo in range(10, 21, 2):
        for hi in range(max(lo, 16), 27, 2):
            X = _features(history["date"], history["temp_forecast"], lo, hi)
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            sse = ((y - X @ b) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, b, lo, hi)
    return best[1:]


def predict(state, history, today) -> float:
    b, lo, hi = state
    X = _features([today["date"]], [today["temp_forecast"]], lo, hi)
    return float(X[0] @ b)
