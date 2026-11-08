"""Do ten changes that are each inside the margin of luck add up to a provable gain?

    python3 scripts/bundle_sim.py     # about 30 seconds

Case A: ten REAL small effects. A copy of the town gets ten kinds of local event, each adding
        1.4 MWh on about 10% of days, and the model can add each event calendar as an input.
Case B: ten pure-noise inputs, each the best of 7 on the scored year, that look better there
        but don't clear the 95% bar: near-misses picked on the same year that tests the bundle.
Case C: the same, but the pieces are picked on the practice year.

First run (MWh; "luck" = one standard deviation of the measured gain):
    A  one real change         gain  0.018  luck 0.021  clears the bar 20% of the time
    A  ten real, bundled       gain  0.181  luck 0.066  clears the bar 89%
    B  ten noise near-misses   gain  0.048  luck 0.030  clears the bar 45%, truly 0.025 worse
    C  picked on practice      gain -0.059  luck 0.033  clears the bar  0%, truly 0.024 worse
Real gains add up while luck grows only with the square root of the number of pieces. But luck
that was picked on the scored year adds up too, so pick the pieces on data the test doesn't use.
"""
import numpy as np
import pandas as pd

import night_sim as S

BETA = 1.4                       # MWh added on each event day
EVENTS = tuple(range(2, 22, 2))  # even ids are event calendars (10% of days) in night_sim.junk


class EventTown(S.Town):
    """The toy town, plus ten real event effects in the load."""

    def __init__(self, seed):
        self.seed = seed
        t, y = S.make_town(seed)
        for jid in EVENTS:
            y = np.round(y + BETA * S.junk(jid, self), 1)
        self.t, self.y = t, y
        onehot = np.eye(7)[S.DOW]
        self.cy = np.concatenate([[0.0], np.cumsum(y)])
        self.cD = np.vstack([np.zeros(7), np.cumsum(onehot, axis=0)])
        centered = pd.Series(y).rolling(7, center=True, min_periods=7).mean().to_numpy()
        gap = np.nan_to_num(y - centered)
        ok = ~np.isnan(centered)
        self.cGap = np.vstack([np.zeros(7), np.cumsum(onehot * (gap * ok)[:, None], axis=0)])
        self.cCnt = np.vstack([np.zeros(7), np.cumsum(onehot * ok[:, None], axis=0)])


def compare(old, new):
    """Gain (positive = better), luck (one standard deviation of the gain), and whether the
    gain clears the old 95% bar (about 1.65 times the luck)."""
    d = new - old
    luck = d.std() / np.sqrt(len(d))
    return -d.mean(), luck, (-d.mean() / luck) >= 1.645


def case_a(worlds=100):
    rows = []
    for u in range(worlds):
        w = EventTown(1000 + u)
        base = S.errors(S.BEST, w, S.VAL)
        for jid in EVENTS:
            rows.append(("one real change", *compare(base, S.errors(dict(S.BEST, junk=(jid,)), w, S.VAL))))
        rows.append(("ten real, bundled", *compare(base, S.errors(dict(S.BEST, junk=EVENTS), w, S.VAL))))
    df = pd.DataFrame(rows, columns=["what", "gain", "luck", "clears"])
    return df.groupby("what", sort=False).mean()


def near_miss_bundle(world, pick_on, first_id, truth_towns):
    """Collect 10 near-misses: the best of 7 noise inputs versus the best model, chosen on
    `pick_on`. When picking on the scored year, keep only ones that look better there but
    don't clear the bar. Then score the bundle of all 10 on the scored year and in truth."""
    base_val = S.errors(S.BEST, world, S.VAL)
    base_pick = S.errors(S.BEST, world, pick_on).mean()
    chosen, next_id = [], first_id
    while len(chosen) < 10:
        ids = list(range(next_id, next_id + 7))
        next_id += 7
        scores = [S.errors(dict(S.BEST, junk=(j,)), world, pick_on).mean() for j in ids]
        if min(scores) >= base_pick:
            continue  # nothing looked better: nothing submitted
        j = ids[int(np.argmin(scores))]
        gain, luck, clears = compare(base_val, S.errors(dict(S.BEST, junk=(j,)), world, S.VAL))
        if pick_on is S.VAL and (clears or gain <= 0):
            continue  # only near-misses: better, but not provably
        chosen.append(j)
    bundle = dict(S.BEST, junk=tuple(chosen))
    gain, luck, clears = compare(base_val, S.errors(bundle, world, S.VAL))
    true_gain = -np.mean([S.errors(bundle, t, S.VAL + S.HOLDOUT).mean()
                          - S.errors(S.BEST, t, S.VAL + S.HOLDOUT).mean() for t in truth_towns])
    return gain, luck, clears, true_gain


def case_bc(pick_on, worlds=40):
    truth_towns = [S.Town(3000 + u) for u in range(20)]
    rows = [near_miss_bundle(S.Town(2000 + i), pick_on, 10_001 + 1000 * i, truth_towns) for i in range(worlds)]
    return pd.DataFrame(rows, columns=["gain", "luck", "clears", "true gain"]).mean()


if __name__ == "__main__":
    print("Case A: ten REAL small effects (100 towns, each scored on its own year)")
    print(case_a().round(3).to_string())
    print("\nCase B: ten noise near-misses picked on the scored year, bundled (40 towns)")
    print(case_bc(S.VAL).round(3).to_string())
    print("\nCase C: ten noise inputs picked on the practice year, bundled (40 towns)")
    print(case_bc(S.PRACTICE).round(3).to_string())
