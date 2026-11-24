"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, prior) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    p = np.asarray(prior, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    cols += [np.maximum(17 - p, 0) + np.maximum(p - 21, 0)]
    return np.column_stack(cols)


def _prior(temps, n=5):
    s = pd.Series(np.asarray(temps, dtype=float))
    return s.shift(1).rolling(n, min_periods=1).mean().bfill().to_numpy()


def fit(history) -> np.ndarray:
    """Least squares on weekday dummies, heating/cooling degrees, lagged-temp extremity."""
    h = history.sort_values("date")
    X = _features(h["date"], h["temp_forecast"], _prior(h["temp_forecast"]))
    return np.linalg.lstsq(X, h["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    h = history.sort_values("date")
    temps = list(h["temp_forecast"].to_numpy(float)) + [float(today["temp_forecast"])]
    X = _features([today["date"]], [temps[-1]], [_prior(temps)[-1]])
    return float(X[0] @ state)
