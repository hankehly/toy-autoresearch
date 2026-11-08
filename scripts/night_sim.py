"""A simulated night: 300 research turns, with a random-tweak robot standing in for the agent.

The town is made up, so we can measure each model's TRUE error: re-live the same three
years in 100 parallel towns (same recipe, fresh weather and noise) and average the misses.
That lets us check every "kept" decision against the truth.

    python3 scripts/night_sim.py      # about 5 minutes; prints the summary, writes night_sim.csv

The robot is dumber than Claude, but the trap is statistical, so it catches any researcher.
Each turn it has one idea and tries 7 versions of it privately, then submits the best
(turn 3 of the real run did exactly this with 7 temperature thresholds). Half its ideas
tweak a setting; half add a new input that sounds useful but is pure noise in this town:
a made-up events calendar or a humidity-like series.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

DAYS = pd.date_range("2023-01-01", "2025-12-31", freq="D")
N = len(DAYS)
DOW = DAYS.dayofweek.to_numpy()
MOY = DAYS.month.to_numpy() - 1
STARTS = pd.date_range("2023-01-01", "2026-01-01", freq="MS")
FIRST = {m: int(DAYS.searchsorted(m)) for m in STARTS}


def months(a, b):
    return list(pd.date_range(a, b, freq="MS"))


PRACTICE = months("2023-07-01", "2024-06-01")  # the agent's own year, after the fix
VAL = months("2024-07-01", "2025-06-01")       # the judge's validation year
HOLDOUT = months("2025-07-01", "2025-12-01")


def make_town(seed):
    """Same recipe as make_data.py. Seed 7 is the real toy town."""
    rng = np.random.default_rng(seed)
    season = 17 - 11 * np.cos(2 * np.pi * (DAYS.dayofyear.to_numpy() - 25) / 365.25)
    spell = np.zeros(N)
    for i in range(1, N):
        spell[i] = 0.7 * spell[i - 1] + rng.normal(0, 1.8)
    temp = season + spell
    temp_forecast = temp + rng.normal(0, 1.0, N)
    heating = 1.8 * np.maximum(0, 16 - temp)
    cooling = 2.5 * np.maximum(0, temp - 22)
    weekend = np.select([DOW == 5, DOW == 6], [-8.0, -12.0], 0.0)
    load = 100 + heating + cooling + weekend + rng.normal(0, 3, N)
    return np.round(temp_forecast, 1), np.round(load, 1)


class Town:
    def __init__(self, seed):
        self.seed = seed
        self.t, self.y = make_town(seed)
        onehot = np.eye(7)[DOW]
        self.cy = np.concatenate([[0.0], np.cumsum(self.y)])
        self.cD = np.vstack([np.zeros(7), np.cumsum(onehot, axis=0)])
        centered = pd.Series(self.y).rolling(7, center=True, min_periods=7).mean().to_numpy()
        gap = np.nan_to_num(self.y - centered)
        ok = ~np.isnan(centered)
        self.cGap = np.vstack([np.zeros(7), np.cumsum(onehot * (gap * ok)[:, None], axis=0)])
        self.cCnt = np.vstack([np.zeros(7), np.cumsum(onehot * ok[:, None], axis=0)])


JUNK = {}


def junk(jid, town):
    """A made-up input with no link to the load. Even ids: an events calendar (10% of days).
    Odd ids: a slow, humidity-like series. Every parallel town gets its own fresh draw."""
    key = (jid, town.seed)
    if key not in JUNK:
        rng = np.random.default_rng([jid, town.seed])
        if jid % 2 == 0:
            JUNK[key] = (rng.random(N) < 0.1).astype(float)
        else:
            JUNK[key] = lfilter([np.sqrt(1 - 0.95 ** 2)], [1, -0.95], rng.normal(size=N))
    return JUNK[key]


def dow_offsets(town, h0, s):
    """How far each weekday sits from its centred week, averaged over the history (as in turn 2)."""
    a, b = h0 + 3, s - 3
    sums, cnts = town.cGap[b] - town.cGap[a], town.cCnt[b] - town.cCnt[a]
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)


def month_effects(resid, idx):
    sums = np.bincount(MOY[idx], weights=resid, minlength=12)
    cnts = np.bincount(MOY[idx], minlength=12)
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)


def run_month(p, town, F, cF, h0, s, e):
    j = np.arange(s, e)
    if p["mode"] == "anchored":  # last week's level + weekday + change in inputs (turn 2's model)
        w = p["win"]
        off = dow_offsets(town, h0, s) if p["dow"] else np.zeros(7)
        i = np.arange(h0 + w, s)
        level = lambda k: (town.cy[k] - town.cy[k - w]) / w - ((town.cD[k] - town.cD[k - w]) / w) @ off
        change = lambda k: F[k] - (cF[k] - cF[k - w]) / w
        X, target = change(i), town.y[i] - off[DOW[i]] - level(i)
        coef = np.linalg.lstsq(X, target, rcond=None)[0] if len(i) > 30 else np.zeros(F.shape[1])
        used = coef.copy()
        used[:2] *= p["shrink"]
        pred = level(j) + off[DOW[j]] + change(j) @ used
        resid = target - X @ coef
    else:  # absolute: load straight from temperature, weekday and any extra inputs
        i = np.arange(h0, s)
        design = lambda k: np.column_stack(
            [np.ones(len(k)), F[k]] + ([np.eye(7)[DOW[k]][:, 1:]] if p["dow"] else []))
        Z = design(i)
        coef = np.linalg.lstsq(Z, town.y[i], rcond=None)[0]
        used = coef.copy()
        used[1:3] *= p["shrink"]
        pred = design(j) @ used
        resid = town.y[i] - Z @ coef
    if p["moy"]:
        pred = pred + month_effects(resid, i)[MOY[j]]
    return pred


def errors(p, town, month_list):
    """Walk forward a month at a time, exactly like the judge. Returns each day's miss."""
    cols = [np.maximum(p["th"] - town.t, 0), np.maximum(town.t - p["tc"], 0)]
    F = np.column_stack(cols + [junk(jid, town) for jid in p["junk"]])
    cF = np.vstack([np.zeros(F.shape[1]), np.cumsum(F, axis=0)])
    out = []
    for m in month_list:
        s, e = FIRST[m], FIRST[m + pd.offsets.MonthBegin(1)]
        h0 = 0 if p["hist"] >= 99 else FIRST.get(m - pd.DateOffset(months=p["hist"]), 0)
        out.append(np.abs(town.y[s:e] - run_month(p, town, F, cF, h0, s, e)))
    return np.concatenate(out)


def answer_key_errors(town, month_list):
    idx = np.concatenate([np.arange(FIRST[m], FIRST[m + pd.offsets.MonthBegin(1)]) for m in month_list])
    t = town.t[idx]
    weekend = np.select([DOW[idx] == 5, DOW[idx] == 6], [-8.0, -12.0], 0.0)
    return np.abs(town.y[idx] - (100 + 1.8 * np.maximum(0, 16 - t) + 2.5 * np.maximum(0, t - 22) + weekend))


# ---------------- the judge's keep rules ----------------
DAY_PICKS = np.random.default_rng(0).integers(0, 365, size=(1000, 365))   # judge.py today
WEEK = np.minimum(np.arange(365) // 7, 51)                                # 52 weeks
WEEK_PICKS = np.random.default_rng(0).integers(0, 52, size=(10000, 52))


def keep_by_days(old, new, k):
    """Today's rule: re-deal single days 1,000 times; keep if the candidate wins 95%."""
    share = float((new[DAY_PICKS].mean(1) < old[DAY_PICKS].mean(1)).mean())
    return share >= 0.95, share


def keep_by_weeks(old, new, k):
    """Fixed rule: re-deal whole weeks; the k-th idea tested tonight must win 1 - 5%/k of re-deals."""
    sums = np.bincount(WEEK, weights=new - old, minlength=52)
    cnts = np.bincount(WEEK, minlength=52).astype(float)
    share = float((sums[WEEK_PICKS].sum(1) / cnts[WEEK_PICKS].sum(1) < 0).mean())
    return share >= 1 - 0.05 / k, share


# ---------------- the robot ----------------
TODAY = dict(mode="anchored", th=18.0, tc=18.0, win=7, hist=99, shrink=1.0, dow=True, moy=False, junk=())  # turn 2
BEST = dict(TODAY, mode="absolute", th=16.0, tc=22.0)  # about as good as this kind of model gets
SETTINGS = ["mode", "th", "tc", "win", "hist", "shrink", "dow", "moy"]


def propose(p, idea, rng, fresh_ids):
    if idea == "new input":
        return [dict(p, junk=p["junk"] + (next(fresh_ids),)) for _ in range(7)]
    if idea == "mode":
        flipped = dict(p, mode="absolute" if p["mode"] == "anchored" else "anchored")
        out = [flipped]
        for _ in range(6):
            th = rng.uniform(8, 24)
            out.append(dict(flipped, th=round(th, 1), tc=round(rng.uniform(max(th, 12), 30), 1)))
        return out
    if idea == "th":
        return [dict(p, th=round(float(np.clip(p["th"] + rng.normal(0, 2.5), 8, p["tc"])), 1)) for _ in range(7)]
    if idea == "tc":
        return [dict(p, tc=round(float(np.clip(p["tc"] + rng.normal(0, 2.5), p["th"], 30)), 1)) for _ in range(7)]
    if idea == "shrink":
        return [dict(p, shrink=round(float(np.clip(p["shrink"] + rng.normal(0, 0.1), 0.5, 1.5)), 2)) for _ in range(7)]
    if idea == "win":
        return [dict(p, win=int(w)) for w in rng.choice([w for w in range(3, 29) if w != p["win"]], 7, replace=False)]
    if idea == "hist":
        return [dict(p, hist=h) for h in [6, 9, 12, 18, 99] if h != p["hist"]]
    return [dict(p, **{idea: not p[idea]})]  # dow, moy: flip


class Lab:
    def __init__(self, universes=100):
        self.town = Town(7)
        self.multiverse = [Town(1000 + u) for u in range(universes)]
        self.cache = {}

    def score(self, p, what):
        key = (tuple(sorted(p.items())), what)
        if key not in self.cache:
            if what == "truth":
                self.cache[key] = float(np.mean([errors(p, t, VAL + HOLDOUT).mean() for t in self.multiverse]))
            else:
                self.cache[key] = errors(p, self.town, {"practice": PRACTICE, "val": VAL, "holdout": HOLDOUT}[what])
        return self.cache[key]

    def night(self, start, private, rule, seed, new_input_share=0.5, turns=300):
        """private: which year the robot tests on before submitting ("val" = today's toy, "practice" = fixed).
        rule: keep_by_days (today) or keep_by_weeks (fixed)."""
        JUNK.clear()
        self.cache = {k: v for k, v in self.cache.items() if not dict(k[0])["junk"]}
        rng = np.random.default_rng(seed)
        fresh_ids = iter(range(10 ** 6))
        champ, k, rows = start, 0, []
        for turn in range(1, turns + 1):
            if rng.random() < new_input_share:
                idea = "new input"
            else:
                idea = str(rng.choice([s for s in SETTINGS if not (s == "win" and champ["mode"] == "absolute")]))
            variants = propose(champ, idea, rng, fresh_ids)
            private_maes = [self.score(v, private).mean() for v in variants]
            best = variants[int(np.argmin(private_maes))]
            submitted = kept = False
            share = truth_change = np.nan
            if min(private_maes) < self.score(champ, private).mean():
                submitted, k = True, k + 1
                kept, share = rule(self.score(champ, "val"), self.score(best, "val"), k)
                if kept:
                    truth_change = self.score(best, "truth") - self.score(champ, "truth")
                    champ = best
            rows.append(dict(turn=turn, idea=idea, submitted=submitted, kept=kept, win_share=share,
                             truth_change=truth_change, val=self.score(champ, "val").mean(),
                             holdout=self.score(champ, "holdout").mean(), truth=self.score(champ, "truth"),
                             inputs=len(champ["junk"]), champion=str(champ)))
        return pd.DataFrame(rows)


SYSTEMS = {
    "today's toy": ("val", keep_by_days),
    "hide answers only": ("practice", keep_by_days),
    "stricter judge only": ("val", keep_by_weeks),
    "both fixes": ("practice", keep_by_weeks),
}
SCENARIOS = {
    "from today's model": TODAY,
    "from the best model": BEST,
}

if __name__ == "__main__":
    lab = Lab()
    key_truth = float(np.mean([answer_key_errors(t, VAL + HOLDOUT).mean() for t in lab.multiverse]))
    print(f"answer key: validation {answer_key_errors(lab.town, VAL).mean():.3f}  "
          f"holdout {answer_key_errors(lab.town, HOLDOUT).mean():.3f}  truth {key_truth:.3f}")
    nights = []
    for scenario, start in SCENARIOS.items():
        s = {w: lab.score(start, w) for w in ("val", "holdout", "truth")}
        print(f"\n{scenario}: start validation {s['val'].mean():.3f}  "
              f"holdout {s['holdout'].mean():.3f}  truth {s['truth']:.3f}")
        for name, (private, rule) in SYSTEMS.items():
            for seed in range(10):
                df = lab.night(start, private, rule, seed)
                df["scenario"], df["system"], df["seed"] = scenario, name, seed
                df["ledger_gain"] = s["val"].mean() - df.val
                df["true_gain"] = s["truth"] - df.truth
                nights.append(df)
        part = pd.concat([n for n in nights if n.scenario.iloc[0] == scenario])
        end, keeps = part[part.turn == 300], part[part.kept]
        print(pd.DataFrame({
            "kept": keeps.groupby("system").size().reindex(SYSTEMS, fill_value=0) / 10,
            "noise inputs kept": end.groupby("system").inputs.mean(),
            "flukes kept": keeps[keeps.truth_change >= 0].groupby("system").size().reindex(SYSTEMS, fill_value=0) / 10,
            "ledger gain": end.groupby("system").ledger_gain.mean(),
            "true gain": end.groupby("system").true_gain.mean(),
            "final holdout": end.groupby("system").holdout.mean(),
        }).reindex(SYSTEMS).round(3).to_string())
    pd.concat(nights).to_csv("night_sim.csv", index=False)
    print("\nAverages over 10 nights of 300 turns each. MWh per day; gain = how much the average miss shrank.")
    print("'flukes' = kept changes that made the true error worse.")
