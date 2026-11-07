"""The exact recipe used to make the data, fed only the forecast (never the actual temperature).
This is about the best any model can do: the leftover miss is weather-forecast error plus pure noise."""


def fit(history):
    return None


def predict(state, history, today):
    t = today["temp_forecast"]
    dow = today["date"].dayofweek
    weekend = -8.0 if dow == 5 else -12.0 if dow == 6 else 0.0
    return 100 + 1.8 * max(0, 16 - t) + 2.5 * max(0, t - 22) + weekend
