"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps):
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    return np.column_stack(cols)


def fit(history):
    """Least squares on log load: day-of-week dummies plus heating/cooling degrees."""
    X = _features(history["date"], history["temp_forecast"])
    return np.linalg.lstsq(X, np.log(history["load"].to_numpy(float)), rcond=None)[0]


def predict(state, history, today):
    X = _features([today["date"]], [today["temp_forecast"]])
    return float(np.exp(X[0] @ state))
