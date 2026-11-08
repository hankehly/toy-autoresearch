# Research brief: tomorrow's electricity use for a small town

You are improving `model.py`, which predicts one day's total electricity use (MWh)
for a made-up town of about 10,000 homes.

## What the model knows when it predicts
- Every past day: its date, its temperature forecast and its actual load.
- Today: the date and today's temperature forecast (in C). Not today's load.

## Your files
- `data.csv`: Jan 2023 - Jun 2024 (date, temp_forecast, load). Explore it freely.
- `model.py`: the current best model. You change this file, and only this file.
- `ledger.tsv`: every experiment so far and its verdict. Read it before you start.
- `check.py`: tells you if model.py crashes. It does not score it.

## How you're judged
A judge you can't see runs model.py one month at a time over Jul 2024 - Jun 2025,
a year you will never see. `fit()` gets every day before the month; `predict()` is
then called once per day. The score is the average miss in MWh (lower is better).

Your change is kept only if it beats the current model in nearly all of 10,000
reshuffles of the judge's 52 weeks. The bar rises with every idea tested: the 1st
must win 95% of reshuffles, the 10th 99.5%, the 100th 99.95%. Small tweaks rarely
clear it, so aim for clear gains.

## Testing before you submit
Use Jul 2023 - Jun 2024 as your practice year: walk forward one month at a time,
exactly like the judge. A practice score is only a guide; the judge's year decides.

## Rules
1. One idea per turn, so we learn what helped.
2. Don't retry an idea the ledger shows was thrown away unless you change it meaningfully.
3. Keep `fit()` and `predict()` working the same way. numpy and pandas only.
   No reading files inside model.py: the judge runs it in a locked sandbox with no files
   and no internet, where that crashes.
4. Keep model.py under 60 lines. When two models tie, the simpler one wins.

## Ideas worth trying (any order)
- Weekends: people use less power.
- Temperature: heaters in the cold, air conditioners in the heat.
- Recent days' loads.
