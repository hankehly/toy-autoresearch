"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, prior) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    p = np.asarray(prior, dtype=float)
    cols = [(dow < 5).astype(float), (dow == 5).astype(float), (dow == 6).astype(float)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    cols += [np.maximum(15 - p, 0), np.maximum(p - 23, 0)]
    return np.column_stack(cols)


def _prior(temps, k=5):
    """Mean of the k forecasts before each day (first days use what exists)."""
    s = pd.Series(np.asarray(temps, dtype=float))
    return s.shift(1).rolling(k, min_periods=1).mean().bfill().to_numpy()


def fit(history) -> np.ndarray:
    temps = history["temp_forecast"]
    X = _features(history["date"], temps, _prior(temps))
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    temps = list(history["temp_forecast"].to_numpy()[-5:])
    prior = float(np.mean(temps)) if temps else float(today["temp_forecast"])
    X = _features([today["date"]], [today["temp_forecast"]], [prior])
    return float(X[0] @ state)
