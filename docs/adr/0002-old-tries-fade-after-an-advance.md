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
