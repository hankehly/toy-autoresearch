# Research brief: tomorrow's electricity use for a small town

You are improving `model.py`, which predicts one day's total electricity use (MWh)
for a made-up town of about 10,000 homes.

## What the model knows when it predicts
- Every past day: its date, its temperature forecast and its actual load.
- Today: the date and today's temperature forecast (in C). Not today's load.

## Your data
`/data` holds {data_first} to {data_last}, read-only: one folder per month with one Parquet file in
it, such as `/data/month={data_first}/part.parquet`. The columns are `date`, `temp_forecast` and
`load`. Explore it freely: `pd.read_parquet("/data")` reads every month as one table, and
adds a `month` column taken from the folder names.

## Your files
- `model.py`: the current best model, the champion. You change this file, and only this file.
- `check.py`: tells you if model.py crashes. It does not score it.
- `practice.py`: scores two models on the practice period, {practice_first} to {practice_last}
  (the newest {practice_months} months of your data), the way the judge scores its own
  period. `python practice.py champion.py model.py` prints both average misses (MWh) and
  the share of 10,000 reshuffles of the practice period's weeks that `model.py` wins.
- `ledger.tsv`: every turn so far. Its numbers are practice numbers: after each turn the
  loop measures the idea against the champion itself, the way `practice.py` does
  (`practice_champion`, `practice`, `practice_wins`). From the judge's period it shows the
  verdict and nothing else:
  - `KEPT`: judged, and it cleared the bar. It became the champion.
  - `thrown away`: judged, and it didn't clear the bar.
  - `dropped on practice`: never judged, because the agent didn't submit it or it won too
    few of the practice reshuffles.
  - `crashed`: `model.py` failed when the loop or the judge ran it.
  - `no change`, `invalid reply`: there was nothing to judge.
  - `advanced`: not a turn and not an idea. New months of data arrived, so your data, the
    practice period and the judge's period all moved forward. The row holds the champion's
    score on the new practice period.

  Only `KEPT` and `thrown away` raise the bar.

  `as_of` is the last month the research could use when the row was measured. Your data
  and the practice period are counted back from it, so they are the same for every row
  with the same `as_of`. Practice scores from rows with an older `as_of` than the newest
  row's were measured on different months: don't compare them with what `practice.py`
  prints today.

  The three lines on top give the champion with its practice score on today's practice
  period, the try count with the bar the next idea must clear at the judge, and the share
  of practice reshuffles an idea must win to be judged.

## How you're judged
A judge you can't see runs model.py one month at a time over the {validation_months} months
after your data ends, which you will never see. `fit()` gets every day before the month;
`predict()` is then called once per day. The score is the average miss in MWh (lower is better).

Your change is kept only if it beats the current model in nearly all of 10,000
reshuffles of the judge's 52 weeks. The bar rises with every idea the judge tests: the
1st must win 95% of reshuffles, the 10th 99.5%, the 100th 99.95%. Small tweaks rarely
clear it, so aim for clear gains.

The bar also eases a little after each `advanced` row. By then the judge's period has
moved on, so an idea judged before the row counts 1/{validation_months} less for each
month it moved, and nothing once all {validation_months} months are new. The second line
of `ledger.tsv` shows the result: the try count, which is how much the judged ideas count
now, and the bar the next idea must clear.

## Each turn, step by step
Your shell runs Python only: `python script.py` or `python -c "..."`.

1. Read `ledger.tsv`, then `model.py`.
2. Copy `model.py` to `champion.py`, so you can compare your change with it.
3. Pick one idea (see Rules and Ideas below) and change `model.py`.
4. Run `python check.py`, and fix `model.py` if it crashes.
5. Run `python practice.py champion.py model.py`. Submit only if `model.py` wins at least
   the share of practice reshuffles named on the third line of `ledger.tsv`. The loop
   measures this itself after your turn, and an idea below that share isn't judged,
   whatever you reply. If yours falls short, you may rework the idea once and run
   `practice.py` again.
6. If it still falls short, leave the idea in `model.py` and set `submit` to false. Don't
   copy `champion.py` back: the loop needs the code to score the idea and log it. An idea
   that isn't submitted is never judged, so it doesn't raise the bar.
7. Finish with the structured reply: the idea you tried, one lesson, the three numbers
   `practice.py` printed for your final `model.py`, and whether to submit it.
   A practice score is only a guide; the judge's period decides.

## Rules
1. One idea per turn, so we learn what helped.
2. Don't retry an idea that was thrown away under the newest `as_of` in the ledger unless
   you change it meaningfully. An idea thrown away under an older `as_of` may be tried
   again as it was: the judge's months have changed since. It counts as a try like any other.
3. If an idea won on practice but the judge threw it away, its gain is either real but too
   small to prove on the judge's period, or luck on the practice period. A near-copy is
   scored on the same months until the next `advanced` row, so it will most likely be
   thrown away too. Try a different idea.
4. Keep `fit()` and `predict()` working the same way. numpy and pandas only.
   Don't read files inside model.py, `/data` included: `fit()` and `predict()` are handed
   the history, and that is all the data a model gets. The judge runs model.py in a locked
   sandbox with no internet, where `/data` stops before the first day it scores.
5. Keep model.py under 60 lines. When two models tie, the simpler one wins.

## Ideas worth trying (any order)
- Weekends: people use less power.
- Temperature: heaters in the cold, air conditioners in the heat.
- Recent days' loads.
