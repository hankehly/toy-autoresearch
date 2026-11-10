# Research brief: tomorrow's electricity use for a small town

You are improving `model.py`, which predicts one day's total electricity use (MWh)
for a made-up town of about 10,000 homes.

## What the model knows when it predicts
- Every past day: its date, its temperature forecast and its actual load.
- Today: the date and today's temperature forecast (in C). Not today's load.

## Your files
- `data.csv`: Jan 2023 - Jun 2024 (date, temp_forecast, load). Explore it freely.
- `model.py`: the current best model, the champion. You change this file, and only this file.
- `check.py`: tells you if model.py crashes. It does not score it.
- `practice.py`: scores two models on the practice year, Jul 2023 - Jun 2024, the way the
  judge scores its own year. `python practice.py champion.py model.py` prints both average
  misses (MWh) and the share of 10,000 reshuffles of the practice year's weeks that
  `model.py` wins.
- `ledger.tsv`: every turn so far. Its numbers are practice numbers: after each turn the
  loop measures the idea against the champion itself, the way `practice.py` does
  (`practice_champion`, `practice`, `practice_wins`). From the judge's year it shows the
  verdict and nothing else:
  - `KEPT`: judged, and it cleared the bar. It became the champion.
  - `thrown away`: judged, and it didn't clear the bar.
  - `dropped on practice`: the agent chose not to submit it, so it was never judged.
  - `crashed`: `model.py` failed when the loop or the judge ran it.
  - `no change`, `invalid reply`: there was nothing to judge.

  Only `KEPT` and `thrown away` raise the bar.

  The two lines on top name the champion and the bar the next idea must clear.

## How you're judged
A judge you can't see runs model.py one month at a time over Jul 2024 - Jun 2025,
a year you will never see. `fit()` gets every day before the month; `predict()` is
then called once per day. The score is the average miss in MWh (lower is better).

Your change is kept only if it beats the current model in nearly all of 10,000
reshuffles of the judge's 52 weeks. The bar rises with every idea the judge tests: the
1st must win 95% of reshuffles, the 10th 99.5%, the 100th 99.95%. Small tweaks rarely
clear it, so aim for clear gains.

## Each turn, step by step
Your shell runs Python only: `python script.py` or `python -c "..."`.

1. Read `ledger.tsv`, then `model.py`.
2. Copy `model.py` to `champion.py`, so you can compare your change with it.
3. Pick one idea (see Rules and Ideas below) and change `model.py`.
4. Run `python check.py`, and fix `model.py` if it crashes.
5. Run `python practice.py champion.py model.py`. Submit only if `model.py` wins at least
   90% of the practice reshuffles. If it doesn't, you may rework the idea once and run
   `practice.py` again.
6. If it still falls short, leave the idea in `model.py` and set `submit` to false. Don't
   copy `champion.py` back: the loop needs the code to score the idea and log it. An idea
   that isn't submitted is never judged, so it doesn't raise the bar.
7. Finish with the structured reply: the idea you tried, one lesson, the three numbers
   `practice.py` printed for your final `model.py`, and whether to submit it.
   A practice score is only a guide; the judge's year decides.

## Rules
1. One idea per turn, so we learn what helped.
2. Don't retry an idea the ledger shows was thrown away unless you change it meaningfully.
3. If an idea won on practice but the judge threw it away, its gain is either real but too
   small to prove on one year, or luck on the practice year. A near-copy is scored on the
   same years, so it will most likely be thrown away too. Try a different idea.
4. Keep `fit()` and `predict()` working the same way. numpy and pandas only.
   No reading files inside model.py: the judge runs it in a locked sandbox with no files
   and no internet, where that crashes.
5. Keep model.py under 60 lines. When two models tie, the simpler one wins.

## Ideas worth trying (any order)
- Weekends: people use less power.
- Temperature: heaters in the cold, air conditioners in the heat.
- Recent days' loads.
