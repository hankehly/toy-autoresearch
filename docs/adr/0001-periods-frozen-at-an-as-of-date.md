---
status: accepted
---

# Periods are frozen at an as-of date and advanced by hand

New data arrives every week, but the four periods (training, practice, validation, holdout)
do not follow it. They are counted back from one stored month, the as-of date, and they stay
where they are until a person advances it. We chose this because the keep rule counts tries
against one validation period and compares every idea with a champion measured on that same
period; if the period moved while ideas were being judged, neither the try count nor the
comparison would mean anything.

## Considered options

- **Follow the newest data on every run.** Always current, but two turns of the same night
  could be judged on different months.
- **Never move the periods.** Simplest, but the agent's data and the validation period grow
  stale while new months pile up unused.
- **Keep the validation period's start and let it grow.** A longer validation period has less
  luck (about 0.049 MWh at 12 months, 0.035 at 24), but the agent never sees recent data and
  judging gets slower. The validation length stays a setting, so this can still be done
  later by raising it.

## Consequences

- Every ledger row records the as-of date it was measured under. Scores from before and
  after an advance are measured on different months and cannot be compared directly.
- An advance moves whole months only, because the judge refits the model once per month.
- Months move one way: holdout, then validation, then practice, then training. A month the
  agent has seen never becomes validation or holdout data again.
- The try count has to say what an old try is worth after an advance. See ADR 0002.
