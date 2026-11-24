"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, lag) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow < 5).astype(float), (dow == 5).astype(float), (dow == 6).astype(float)]
    cols += [np.maximum(16 - t, 0), np.maximum(t - 22, 0), np.abs(lag - 19)]
    return np.column_stack(cols)


def _lag(temps) -> np.ndarray:
    """Mean of the previous 5 forecasts (expanding at the start)."""
    s = pd.Series(np.asarray(temps, dtype=float))
    return s.shift(1).rolling(5, min_periods=1).mean().bfill().to_numpy()


def fit(history) -> np.ndarray:
    X = _features(history["date"], history["temp_forecast"], _lag(history["temp_forecast"]))
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    lag = history["temp_forecast"].to_numpy(float)[-5:].mean()
    X = _features([today["date"]], [today["temp_forecast"]], np.array([lag]))
    return float(X[0] @ state)
