"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, lag) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow < 5).astype(float), (dow == 5).astype(float), (dow == 6).astype(float)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0), np.abs(np.asarray(lag, float) - 19)]
    return np.column_stack(cols)


def _lag(temps) -> np.ndarray:
    """Mean of the prior 5 days' temperature (own temp where no history)."""
    s = pd.Series(np.asarray(temps, dtype=float))
    return s.shift(1).rolling(5, min_periods=1).mean().fillna(s).to_numpy()


def fit(history) -> np.ndarray:
    """Least squares: weekday/Sat/Sun dummies, heating/cooling degrees, lagged temp extremity."""
    t = history["temp_forecast"].to_numpy(float)
    X = _features(history["date"], t, _lag(t))
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    t = history["temp_forecast"].to_numpy(float)[-5:]
    lag = t.mean() if len(t) else today["temp_forecast"]
    X = _features([today["date"]], [today["temp_forecast"]], [lag])
    return float(X[0] @ state)
