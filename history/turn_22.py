"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    return np.column_stack(cols)


def fit(history) -> np.ndarray:
    """Huber-weighted regression on day-of-week dummies plus heating/cooling degrees."""
    X = _features(history["date"], history["temp_forecast"])
    y = history["load"].to_numpy(float)
    w = np.ones(len(y))
    for _ in range(5):  # Huber-style reweighting damps outlier days
        b = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        r = y - X @ b
        c = 1.5 * 1.4826 * np.median(np.abs(r))
        w = np.sqrt(np.minimum(1.0, c / np.maximum(np.abs(r), 1e-9)))
    return b


def predict(state, history, today) -> float:
    X = _features([today["date"]], [today["temp_forecast"]])
    return float(X[0] @ state)
