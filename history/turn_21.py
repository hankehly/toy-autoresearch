"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def _ewm(temps, today_t):
    """Exponentially weighted mean of strictly prior temperatures (alpha .3)."""
    s = pd.Series(np.asarray(temps, float))
    return s.ewm(alpha=0.3, adjust=False).mean().shift(1).bfill().to_numpy()


def _features(dates, temps, lag) -> np.ndarray:
    dow = np.asarray(pd.DatetimeIndex(dates).dayofweek)
    t = np.asarray(temps, dtype=float)
    cols = [(dow < 5).astype(float), (dow == 5).astype(float), (dow == 6).astype(float)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0), np.abs(lag - 19)]
    return np.column_stack(cols)


def fit(history) -> np.ndarray:
    t = history["temp_forecast"].to_numpy(float)
    X = _features(history["date"], t, _ewm(t, None))
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    t = np.append(history["temp_forecast"].to_numpy(float), today["temp_forecast"])
    lag = _ewm(t, None)[-1:]
    X = _features([today["date"]], [today["temp_forecast"]], lag)
    return float(X[0] @ state)
