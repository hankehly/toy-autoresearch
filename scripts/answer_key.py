"""The exact recipe used to make the data, fed only the forecast (never the actual temperature).

This is about the best any model can do: the leftover miss is weather-forecast error plus
pure noise.
"""


def fit(history) -> None:
    """Learn nothing: the recipe is already written into predict()."""
    return None


def predict(state, history, today) -> float:
    """Guess today's load from today's forecast temperature and weekday alone.

    Parameters
    ----------
    state : None
        What fit() returned: nothing.
    history : pandas.DataFrame
        Every day before today. Not used.
    today : dict
        Today's columns except `load`: its `date` and `temp_forecast`.

    Returns
    -------
    float
        The load in MWh: 100, plus heating below 16 degrees, plus cooling above 22 degrees,
        less on Saturday and less again on Sunday.
    """
    t = today["temp_forecast"]
    dow = today["date"].dayofweek
    weekend = -8.0 if dow == 5 else -12.0 if dow == 6 else 0.0
    return 100 + 1.8 * max(0, 16 - t) + 2.5 * max(0, t - 22) + weekend
