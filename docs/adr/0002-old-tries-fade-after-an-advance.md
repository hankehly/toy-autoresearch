---
status: accepted
---

# Old tries fade as their validation months leave

The bar an idea must clear rises with the try count, because testing many ideas on the same
validation months lets one through by luck. After an advance some of those months are
replaced (ADR 0001), so we had to decide what an old try is still worth. A try now counts in
proportion to how many of the months it was judged on are still in the validation period:
with a 12-month validation period it counts 11/12 after a one-month advance, 6/12 after six
months and nothing after twelve.

Example: 8 tries, then a one-month advance. The try count becomes 8 x 11/12 = 7.33, and the
next idea must win 99.40% of reshuffles instead of 99.44%.

## Considered options

- **Keep the count.** The bar only ever rises. After 100 tries the next idea must win 99.95%
  of reshuffles for good, even once every validation month is new.
- **Reset the count at each advance.** With monthly advances, 11 of the 12 validation months
  are unchanged, so the bar would drop to 95% on nearly the same data. Without a rising bar,
  simulated nights kept about 9 lucky ideas each (CLAUDE.md, finding 2).

## Consequences

- The try count is no longer a whole number.
- This rule is our own, not one taken from the published work in `docs/research.md`. It is
  checked by simulation in `scripts/night_sim.py`, which mirrors the judge's rules.
- The check (2026-11-23, `uv run scripts/night_sim.py advance`; CLAUDE.md, finding 8): 300
  ideas over 12 nights a month apart, 30 runs. Letting tries fade kept 0.03 to 0.10 lucky
  ideas per run, about as many as keeping the count (0.03 to 0.10). Resetting the count
  kept 0.43 to 0.60.
- In the first year the rule is close to keeping the count, because new tries arrive faster
  than old ones fade: after 300 ideas the bar is 99.95% against 99.98%. The two part later.
  At a steady pace the fading count levels off, where a kept count grows for good, and when
  the research pauses the fading bar comes back down.
- An idea thrown away under an older as-of date may be tried again as it was (rule 2 of the
  brief, decided 2026-11-23). The retry is a try like any other. In the same simulation a
  retry was about as likely to be a lucky keep as a new idea: over 200 runs, retries were
  27% of the turns and 10 of the 41 lucky keeps.
