"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, past) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0), np.abs(np.asarray(past, float) - 19)]
    return np.column_stack(cols)


def fit(history) -> np.ndarray:
    """Least squares on day-of-week dummies plus heating/cooling degrees."""
    t = history["temp_forecast"]
    past = t.shift(1).rolling(7, min_periods=1).mean().fillna(t.mean())
    X = _features(history["date"], t, past)
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    past = history["temp_forecast"].tail(7).mean()
    X = _features([today["date"]], [today["temp_forecast"]], [past])
    return float(X[0] @ state)
