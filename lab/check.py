"""Crash check only: runs model.py on the last two months in /data. It does NOT score it."""
from pathlib import Path

import numpy as np
import pandas as pd

import model

pd.options.mode.copy_on_write = True  # as in the judge's sandbox: model.py gets its history without a copy being made

files = sorted(Path("/data").glob("month=*/part.parquet"))  # one file per month, oldest first
data = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
n = 0
for start in pd.date_range(end=data.date.max(), periods=2, freq="MS"):
    month = data[(data.date >= start) & (data.date < start + pd.offsets.MonthBegin(1))]
    state = model.fit(data.iloc[:month.index[0]])  # every row before the month
    for i, day in zip(month.index, month.to_dict("records"), strict=True):
        today = {column: value for column, value in day.items() if column != "load"}
        pred = float(model.predict(state, data.iloc[:i], today))  # every row before today
        assert np.isfinite(pred), f"prediction for {day['date']:%Y-%m-%d} is not a number"
        n += 1
print(f"OK: model.py ran on {n} days without crashing.")
