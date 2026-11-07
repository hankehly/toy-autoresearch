"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, prev):
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    p = np.asarray(prev, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    cols += [np.maximum(17 - p, 0), np.maximum(p - 21, 0)]
    return np.column_stack(cols)


def fit(history):
    """Least squares on weekday dummies plus heating/cooling degrees (today and yesterday)."""
    t = history["temp_forecast"].to_numpy(float)
    prev = np.r_[t[0], t[:-1]]
    X = _features(history["date"], t, prev)
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today):
    prev = history["temp_forecast"].iloc[-1] if len(history) else today["temp_forecast"]
    X = _features([today["date"]], [today["temp_forecast"]], [prev])
    return float(X[0] @ state)
