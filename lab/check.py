"""Crash check only: runs model.py on the last two months of data.csv. It does NOT score it."""
import numpy as np
import pandas as pd

import model

data = pd.read_csv("data.csv", parse_dates=["date"])
n = 0
for start in pd.date_range(end=data.date.max(), periods=2, freq="MS"):
    state = model.fit(data[data.date < start].copy())
    month = data[(data.date >= start) & (data.date < start + pd.offsets.MonthBegin(1))]
    for day in month.itertuples():
        today = {"date": day.date, "temp_forecast": day.temp_forecast}
        pred = float(model.predict(state, data[data.date < day.date].copy(), today))
        assert np.isfinite(pred), f"prediction for {day.date:%Y-%m-%d} is not a number"
        n += 1
print(f"OK: model.py ran on {n} days without crashing.")
