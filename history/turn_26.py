"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def _features(dates, temps) -> np.ndarray:
    idx = pd.DatetimeIndex(dates)
    dow = np.asarray(idx.dayofweek)
    hol = np.asarray(idx.isin(USFederalHolidayCalendar().holidays("2020", "2035")))
    t = np.asarray(temps, dtype=float)
    cols = [((dow < 5) & ~hol).astype(float), (dow == 5).astype(float),
            ((dow == 6) | hol).astype(float)]
    cols += [np.maximum(17 - t, 0), np.maximum(t - 21, 0)]
    return np.column_stack(cols)


def fit(history) -> np.ndarray:
    """Least squares on pooled weekday/Sat/Sun-or-holiday dummies plus heating/cooling degrees."""
    X = _features(history["date"], history["temp_forecast"])
    return np.linalg.lstsq(X, history["load"].to_numpy(float), rcond=None)[0]


def predict(state, history, today) -> float:
    X = _features([today["date"]], [today["temp_forecast"]])
    return float(X[0] @ state)
