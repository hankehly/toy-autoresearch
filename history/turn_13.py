"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    return np.column_stack(cols)


def _eff(temps) -> np.ndarray:
    """Effective temperature: today's forecast blended with the prior 3 days."""
    t = pd.Series(np.asarray(temps, dtype=float))
    prev = t.shift(1).rolling(3, min_periods=1).mean().fillna(t)
    return (0.3 * t + 0.7 * prev).to_numpy()


def fit(history) -> np.ndarray:
    """Least squares on day-of-week dummies plus heating/cooling degrees."""
    X = _features(history["date"], _eff(history["temp_forecast"]))
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    t = list(history["temp_forecast"].to_numpy(float)[-3:]) + [float(today["temp_forecast"])]
    X = _features([today["date"]], _eff(t)[-1:])
    return float(X[0] @ state)
