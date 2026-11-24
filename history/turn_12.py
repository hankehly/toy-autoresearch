"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _features(dates, temps, prev) -> np.ndarray:
    """prev: mean temperature forecast of the 3 days before each date."""
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    p = np.asarray(prev, dtype=float)
    cols = [(dow == d).astype(float) for d in range(7)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    cols += [np.maximum(17 - p, 0), np.maximum(p - 21, 0)]
    return np.column_stack(cols)


def _prev(temps) -> np.ndarray:
    s = pd.Series(np.asarray(temps, dtype=float))
    return s.shift(1).rolling(3, min_periods=1).mean().bfill().to_numpy()


def fit(history) -> np.ndarray:
    """Least squares on weekday dummies, heating/cooling degrees today and of prior 3 days."""
    h = history.sort_values("date")
    X = _features(h["date"], h["temp_forecast"], _prev(h["temp_forecast"]))
    return np.linalg.lstsq(X, h["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    h = history.sort_values("date")
    p = h["temp_forecast"].to_numpy(float)[-3:].mean()
    X = _features([today["date"]], [today["temp_forecast"]], [p])
    return float(X[0] @ state)
