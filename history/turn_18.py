"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, prev) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    cols.append(t - np.asarray(prev, dtype=float))
    return np.column_stack(cols)


def fit(history) -> np.ndarray:
    """Least squares on day-of-week dummies plus heating/cooling degrees."""
    X = _features(history["date"], history["temp_forecast"], history["temp_forecast"].shift(1).bfill())
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    X = _features([today["date"]], [today["temp_forecast"]], [history["temp_forecast"].iloc[-1]])
    return float(X[0] @ state)
