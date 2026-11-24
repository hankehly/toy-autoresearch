"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    return np.column_stack(cols)


def _smooth(temps) -> np.ndarray:
    """Average each day's forecast with the previous day's."""
    t = np.asarray(temps, dtype=float)
    return np.concatenate([t[:1], (t[1:] + t[:-1]) / 2])


def fit(history) -> np.ndarray:
    X = _features(history["date"], _smooth(history["temp_forecast"]))
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    t = [history["temp_forecast"].iloc[-1], today["temp_forecast"]]
    X = _features([today["date"]], _smooth(t)[1:])
    return float(X[0] @ state)
