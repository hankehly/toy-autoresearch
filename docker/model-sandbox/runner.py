"""The model sandbox's only program. It loads model.py and answers the judge's messages.

It starts by loading the month folders the judge mounted at /data: every day before the
first scored one. Then each message is one line of JSON text, arriving on stdin:
  fit      learn from every day so far. Reply {"ok": true}.
  predict  today: its date and every other column except the load. Reply {"guess": <load in MWh>}.
  reveal   these days are now in the past (every column, the load included). No reply.

The sandbox knows only the months it started with and what the messages have told it. The
judge reveals a day's load only after this program has sent back its guess for that day.
"""
import json
import sys
from pathlib import Path

import pandas as pd

# Copy-on-write: model.py is handed the past without a copy being made. If it writes to that
# table, pandas copies only the part it writes to, so the next call still sees the original.
pd.options.mode.copy_on_write = True

to_judge = sys.stdout
sys.stdout = sys.stderr  # anything model.py prints goes to the log, never to the judge

import model  # noqa: E402  (imported only now, so even its prints at import time go to the log)

files = sorted(Path("/data").glob("month=*/part.parquet"))  # one file per month, oldest first
past = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)  # every day so far, as one table
state = None  # whatever model.fit() returned

for line in sys.stdin:
    message = json.loads(line)

    if message["op"] == "reveal":
        days = pd.DataFrame(message["days"])
        days["date"] = pd.to_datetime(days["date"])
        past = pd.concat([past, days], ignore_index=True)

    elif message["op"] == "fit":
        state = model.fit(past.copy(deep=False))  # deep=False: a new table on the same memory, not a copy
        print(json.dumps({"ok": True}), file=to_judge, flush=True)

    elif message["op"] == "predict":
        today = message["today"]
        today["date"] = pd.Timestamp(today["date"])
        guess = float(model.predict(state, past.copy(deep=False), today))
        print(json.dumps({"guess": guess}), file=to_judge, flush=True)
