# Research brief: tomorrow's electricity use for a small town

You are improving `model.py`, which predicts one day's total electricity use (MWh)
for a made-up town of about 10,000 homes.

## What the model knows when it predicts
- Every past day: its date, its temperature forecast and its actual load.
- Today: the date and today's temperature forecast (in C). Not today's load.

## Your files
- `data.csv`: Jan 2023 - Jun 2024 (date, temp_forecast, load). Explore it freely.
- `model.py`: the current best model. You change this file, and only this file.
- `ledger.tsv`: every experiment so far and its verdict. "no change" means the idea was
  dropped after its practice test. Its scores are from the judge's year, which has more
  history to learn from, so your practice scores will usually be a little higher.
- `check.py`: tells you if model.py crashes. It does not score it.

## How you're judged
A judge you can't see runs model.py one month at a time over Jul 2024 - Jun 2025,
a year you will never see. `fit()` gets every day before the month; `predict()` is
then called once per day. The score is the average miss in MWh (lower is better).

Your change is kept only if it beats the current model in nearly all of 10,000
reshuffles of the judge's 52 weeks. The bar rises with every idea tested: the 1st
must win 95% of reshuffles, the 10th 99.5%, the 100th 99.95%. Small tweaks rarely
clear it, so aim for clear gains.

## Each turn, step by step
Your shell runs Python only: `python script.py` or `python -c "..."`.

1. Read `ledger.tsv`, then `model.py`.
2. Copy `model.py` to `champion.py`, so you can score it and go back to it.
3. Write a script that scores a model on the practice year, Jul 2023 - Jun 2024: walk
   forward one month at a time, exactly like the judge (`check.py` does this for two
   months), and print the average miss. Score `champion.py`.
4. Pick one idea (see Rules and Ideas below) and change `model.py`.
5. Run `python check.py`, and fix `model.py` if it crashes.
6. Score `model.py`. If it isn't clearly better than `champion.py`, you may rework the
   idea. If it still isn't, copy `champion.py` back over `model.py`: a turn that changes
   nothing doesn't count as a try, so it doesn't raise the bar.
7. Reply with what you tried, both practice scores, and whether you're submitting it.
   A practice score is only a guide; the judge's year decides.

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
