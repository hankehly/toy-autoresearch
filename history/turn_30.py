"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, prior) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow < 5).astype(float), (dow == 5).astype(float), (dow == 6).astype(float)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0), np.abs(np.asarray(prior, float) - 19)]
    return np.column_stack(cols)


def _prior(temps, k=5):
    s = pd.Series(np.asarray(temps, float))
    return s.shift(1).rolling(k, min_periods=1).mean().bfill().to_numpy()


def fit(history):
    h = history.sort_values("date")
    t = h["temp_forecast"].to_numpy(float)
    X = _features(h["date"], t, _prior(t))
    y = h["load"].to_numpy(float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, float(np.mean((y - X @ b)[-28:]))


def predict(state, history, today) -> float:
    b, lvl = state
    h = history.sort_values("date")
    t = h["temp_forecast"].to_numpy(float)[-5:]
    p = t.mean() if len(t) else float(today["temp_forecast"])
    X = _features([today["date"]], [today["temp_forecast"]], [p])
    return float(X[0] @ b + 0.5 * lvl)
