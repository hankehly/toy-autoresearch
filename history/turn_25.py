"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd

K = 14


def _features(dates, temps, prior) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow < 5).astype(float), (dow == 5).astype(float), (dow == 6).astype(float)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0), np.abs(np.asarray(prior) - 19)]
    return np.column_stack(cols)


def _prior(temps) -> np.ndarray:
    s = pd.Series(np.asarray(temps, dtype=float))
    return s.shift(1).rolling(K, min_periods=1).mean().bfill().to_numpy()


def fit(history) -> np.ndarray:
    X = _features(history["date"], history["temp_forecast"], _prior(history["temp_forecast"]))
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    p = history["temp_forecast"].to_numpy(float)[-K:].mean()
    X = _features([today["date"]], [today["temp_forecast"]], [p])
    return float(X[0] @ state)
