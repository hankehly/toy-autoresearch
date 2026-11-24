"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, prev) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    p = np.asarray(prev, dtype=float)  # extreme of prior 3 days: (max, min)
    cols = [(dow < 5).astype(float), (dow == 5).astype(float), (dow == 6).astype(float)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    cols += [np.maximum(p[:, 0] - 24, 0), np.maximum(14 - p[:, 1], 0)]
    return np.column_stack(cols)


def _prev(temps):
    s = pd.Series(np.asarray(temps, dtype=float))
    a = s.shift(1).rolling(3, min_periods=1)
    return np.column_stack([a.max().fillna(s.iloc[0]), a.min().fillna(s.iloc[0])])


def fit(history) -> np.ndarray:
    t = history["temp_forecast"].to_numpy(float)
    X = _features(history["date"], t, _prev(t))
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    t = np.append(history["temp_forecast"].to_numpy(float)[-3:], today["temp_forecast"])
    X = _features([today["date"]], [today["temp_forecast"]], _prev(t)[-1:])
    return float(X[0] @ state)
