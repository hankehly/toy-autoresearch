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
    """Least squares on weekday dummies + degrees; AR(1) coefficient on residuals."""
    X = _features(history["date"], history["temp_forecast"])
    y = history["load"].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    phi = float(r[1:] @ r[:-1] / (r[:-1] @ r[:-1]))
    return beta, phi


def predict(state, history, today):
    beta, phi = state
    X = _features([today["date"]], [today["temp_forecast"]])
    pred = float(X[0] @ beta)
    last = history.iloc[-1]
    if (pd.Timestamp(today["date"]) - pd.Timestamp(last["date"])).days == 1:
        Xl = _features([last["date"]], [last["temp_forecast"]])
        pred += phi * (float(last["load"]) - float(Xl[0] @ beta))
    return pred
