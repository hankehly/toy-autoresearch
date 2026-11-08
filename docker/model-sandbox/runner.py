"""The model sandbox's only program. It loads model.py and answers the judge's messages.

Each message is one line of JSON text, arriving on stdin:
  reveal   these days are now in the past (date, temp_forecast, load). No reply.
  fit      learn from every day revealed so far. Reply {"ok": true}.
  predict  today's date and temperature forecast. Reply {"guess": <load in MWh>}.

The sandbox knows only what the messages have told it. The judge reveals a day's load
only after this program has sent back its guess for that day.
"""
import json
import sys

import pandas as pd

to_judge = sys.stdout
sys.stdout = sys.stderr  # anything model.py prints goes to the log, never to the judge

import model  # imported only now, so even its prints at import time go to the log

past = None   # every day revealed so far, as one table
state = None  # whatever model.fit() returned

for line in sys.stdin:
    message = json.loads(line)

    if message["op"] == "reveal":
        days = pd.DataFrame(message["days"])
        days["date"] = pd.to_datetime(days["date"])
        past = days if past is None else pd.concat([past, days], ignore_index=True)

    elif message["op"] == "fit":
        state = model.fit(past.copy())
        print(json.dumps({"ok": True}), file=to_judge, flush=True)

    elif message["op"] == "predict":
        today = {"date": pd.Timestamp(message["date"]), "temp_forecast": message["temp_forecast"]}
        guess = float(model.predict(state, past.copy(), today))
        print(json.dumps({"guess": guess}), file=to_judge, flush=True)
