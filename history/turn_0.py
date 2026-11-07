"""Predict one day's total electricity use (MWh) for the town."""
import numpy as np
import pandas as pd


def fit(history):
    """history: every day before the month being predicted (date, temp_forecast, load).
    Return anything predict() needs."""
    return None


def predict(state, history, today):
    """today: {"date": Timestamp, "temp_forecast": float}. history: every day before today.
    Return the predicted load in MWh."""
    return history["load"].tail(7).mean()
