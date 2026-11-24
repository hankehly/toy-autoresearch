"""A simulated batch: 300 research turns, with a random-tweak robot standing in for the agent.

The town is made up, so we can measure each model's TRUE error: re-live the same years in
100 parallel towns (same recipe, fresh weather and noise) and average the misses.
That lets us check every "kept" decision against the truth.

    uv run scripts/loop_sim.py            # about 5 minutes; prints the summary, writes loop_sim.csv
    uv run scripts/loop_sim.py advance    # about 16 minutes; prints the summary, writes loop_sim_advance.csv

The robot is dumber than Claude, but the trap is statistical, so it catches any researcher.
Each turn it has one idea and tries 7 versions of it privately, then submits the best
(turn 3 of the real run did exactly this with 7 temperature thresholds). Half its ideas
tweak a setting; half add a new input that sounds useful but is pure noise in this town:
a made-up events calendar or a humidity-like series.

With no argument, every batch is judged on the real toy town under one as-of date, and four
systems are compared: the old toy, each of its two fixes alone, and both together.

`advance` asks what a try should still count once the as-of date has advanced
(docs/adr/0002). It spends the same 300 ideas in two ways: in one batch, and in 12 batches
of 25 with a one-month advance between them. The 12 batches are run under three rules for
the try count: keep it, reset it, or let old tries fade, which is the loop's rule. They are
then run again with retries: some turns try again an idea that was thrown away under an
older as-of date, which the brief allows. Every seed is judged on a town of its own, so
that no two seeds share the luck of one town's validation months.
"""

import itertools
import sys

import numpy as np
import pandas as pd
from scipy.signal import lfilter

import make_data  # its town() is the recipe: the simulated towns use it, not a copy of it

# The periods are counted back from an as-of date, as in judge/judge.py: the same three lengths, in months.
HOLDOUT_MONTHS, VALIDATION_MONTHS, PRACTICE_MONTHS = 6, 12, 12
START_AS_OF = pd.Period("2025-12", "M")  # the as-of date every run starts under: a fresh checkout's

# The three years the toy began with, and one year of its pretend future: enough months
# for 12 batches with a one-month advance between them.
DAYS = pd.date_range("2023-01-01", "2026-12-31", freq="D")
N = len(DAYS)
DOW = DAYS.dayofweek.to_numpy()
MOY = DAYS.month.to_numpy() - 1
# Where each month starts in DAYS. The month after the last one is here too: it is where the last one ends.
FIRST = {
    month: int(DAYS.searchsorted(month.start_time))
    for month in pd.period_range("2023-01", "2027-01", freq="M")
}


def periods(as_of) -> dict[str, list[pd.Period]]:
    """Count the periods back from an as-of date, as periods() in judge/judge.py does.

    Parameters
    ----------
    as_of : pd.Period
        The as-of date: a month.

    Returns
    -------
    dict[str, list[pd.Period]]
        The months of "holdout", "val" and "practice", oldest first. The holdout period ends
        at the as-of date, the validation period ends where the holdout period starts, and
        the practice period ends where the validation period starts.
    """
    found, last = {}, as_of
    for name, length in [
        ("holdout", HOLDOUT_MONTHS),
        ("val", VALIDATION_MONTHS),
        ("practice", PRACTICE_MONTHS),
    ]:
        found[name] = list(pd.period_range(end=last, periods=length, freq="M"))
        last -= length  # the next period back ends just before this one starts
    return found


# The periods of the as-of date a run starts under. bundle_sim.py scores on them.
PRACTICE, VAL, HOLDOUT = (periods(START_AS_OF)[name] for name in ("practice", "val", "holdout"))


def make_town(seed, later_seed=None) -> tuple[np.ndarray, np.ndarray]:
    """Make a town with make_data.py's recipe, drawn the way make_data.py draws the real one.

    The days the toy began with come from one generator, and the days after them from a
    second, as one run to the end of the pretend future. Seeds 7 and 8 make the real toy town.

    Parameters
    ----------
    seed : int
        The generator for the days the toy began with.
    later_seed : int, optional
        The generator for the days after them. Left out, it is seed + 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The temperature forecast (C) and the load (MWh) of each day in DAYS.
    """
    # the same two runs of days as make() in make_data.py draws
    began_with = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    future = pd.date_range("2026-01-01", "2042-12-31", freq="D")
    first, spell = make_data.town(began_with, np.random.default_rng(seed))
    later, _ = make_data.town(
        future, np.random.default_rng(seed + 1 if later_seed is None else later_seed), spell
    )
    # the simulation needs only the first of the future's years
    days = pd.concat([first, later]).iloc[:N]
    return days.temp_forecast.to_numpy(), days.load.to_numpy()


class Town:
    """One town's four years: its forecast temperatures (t) and loads (y).

    The other fields are running totals of the loads, kept so that a model can be scored on
    any stretch of days without adding them up again.
    """

    def __init__(self, seed, later_seed=None) -> None:
        self.seed = seed
        self.t, self.y = make_town(seed, later_seed)
        onehot = np.eye(7)[DOW]
        self.cy = np.concatenate([[0.0], np.cumsum(self.y)])
        self.cD = np.vstack([np.zeros(7), np.cumsum(onehot, axis=0)])
        centered = pd.Series(self.y).rolling(7, center=True, min_periods=7).mean().to_numpy()
        gap = np.nan_to_num(self.y - centered)
        ok = ~np.isnan(centered)
        self.cGap = np.vstack([np.zeros(7), np.cumsum(onehot * (gap * ok)[:, None], axis=0)])
        self.cCnt = np.vstack([np.zeros(7), np.cumsum(onehot * ok[:, None], axis=0)])


JUNK = {}


def junk(jid, town) -> np.ndarray:
    """A made-up input with no link to the load.

    Even ids: an events calendar (10% of days). Odd ids: a slow, humidity-like series.
    Every parallel town gets its own fresh draw.
    """
    key = (jid, town.seed)
    if key not in JUNK:
        rng = np.random.default_rng([jid, town.seed])
        if jid % 2 == 0:
            JUNK[key] = (rng.random(N) < 0.1).astype(float)
        else:
            JUNK[key] = lfilter([np.sqrt(1 - 0.95**2)], [1, -0.95], rng.normal(size=N))
    return JUNK[key]


def dow_offsets(town, h0, s) -> np.ndarray:
    """How far each weekday sits from its centred week, averaged over the history (as in turn 2)."""
    a, b = h0 + 3, s - 3
    sums, cnts = town.cGap[b] - town.cGap[a], town.cCnt[b] - town.cCnt[a]
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)


def month_effects(resid, idx) -> np.ndarray:
    """The average leftover miss in each of the 12 calendar months (0 for a month with no days).

    Parameters
    ----------
    resid : numpy.ndarray
        What the fitted model still missed on each training day.
    idx : numpy.ndarray
        Which day each of those misses belongs to.
    """
    sums = np.bincount(MOY[idx], weights=resid, minlength=12)
    cnts = np.bincount(MOY[idx], minlength=12)
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)


def run_month(p, town, F, cF, h0, s, e) -> np.ndarray:
    """Fit a model on the days before one month and guess that month's loads.

    Parameters
    ----------
    p : dict
        The model's settings, such as TODAY or BEST.
    town : Town
        The town to fit and guess in.
    F : numpy.ndarray
        The model's inputs, one row per day: heating, cooling and any extra inputs.
    cF : numpy.ndarray
        The running totals of F.
    h0 : int
        The first day of the history the model may fit on.
    s : int
        The first day of the month.
    e : int
        The day after the month's last.

    Returns
    -------
    numpy.ndarray
        The guessed load for each day of the month.
    """
    j = np.arange(s, e)
    if p["mode"] == "anchored":  # last week's level + weekday + change in inputs (turn 2's model)
        w = p["win"]
        off = dow_offsets(town, h0, s) if p["dow"] else np.zeros(7)
        i = np.arange(h0 + w, s)

        def level(k: np.ndarray) -> np.ndarray:
            return (town.cy[k] - town.cy[k - w]) / w - ((town.cD[k] - town.cD[k - w]) / w) @ off

        def change(k: np.ndarray) -> np.ndarray:
            return F[k] - (cF[k] - cF[k - w]) / w

        X, target = change(i), town.y[i] - off[DOW[i]] - level(i)
        coef = np.linalg.lstsq(X, target, rcond=None)[0] if len(i) > 30 else np.zeros(F.shape[1])
        used = coef.copy()
        used[:2] *= p["shrink"]
        pred = level(j) + off[DOW[j]] + change(j) @ used
        resid = target - X @ coef
    else:  # absolute: load straight from temperature, weekday and any extra inputs
        i = np.arange(h0, s)

        def design(k: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [np.ones(len(k)), F[k]] + ([np.eye(7)[DOW[k]][:, 1:]] if p["dow"] else [])
            )

        Z = design(i)
        coef = np.linalg.lstsq(Z, town.y[i], rcond=None)[0]
        used = coef.copy()
        used[1:3] *= p["shrink"]
        pred = design(j) @ used
        resid = town.y[i] - Z @ coef
    if p["moy"]:
        pred = pred + month_effects(resid, i)[MOY[j]]
    return pred


def errors(p, town, month_list) -> np.ndarray:
    """Walk forward a month at a time, exactly like the judge. Returns each day's miss."""
    cols = [np.maximum(p["th"] - town.t, 0), np.maximum(town.t - p["tc"], 0)]
    F = np.column_stack(cols + [junk(jid, town) for jid in p["junk"]])
    cF = np.vstack([np.zeros(F.shape[1]), np.cumsum(F, axis=0)])
    out = []
    for m in month_list:
        s, e = FIRST[m], FIRST[m + 1]
        h0 = 0 if p["hist"] >= 99 else FIRST.get(m - p["hist"], 0)
        out.append(np.abs(town.y[s:e] - run_month(p, town, F, cF, h0, s, e)))
    return np.concatenate(out)


def answer_key_errors(town, month_list) -> np.ndarray:
    """Each day's miss, over these months, of the recipe that made the data."""
    idx = np.concatenate([np.arange(FIRST[m], FIRST[m + 1]) for m in month_list])
    t = town.t[idx]
    weekend = np.select([DOW[idx] == 5, DOW[idx] == 6], [-8.0, -12.0], 0.0)
    return np.abs(
        town.y[idx] - (100 + 1.8 * np.maximum(0, 16 - t) + 2.5 * np.maximum(0, t - 22) + weekend)
    )


# ---------------- the judge's keep rules ----------------
DAY_PICKS = np.random.default_rng(0).integers(0, 365, size=(1000, 365))  # the old judge.py
# judge.py's re-deals of the 52 weeks
WEEK_PICKS = np.random.default_rng(0).integers(0, 52, size=(10000, 52))


def bar(number) -> float:
    """Work out the share of re-deals an idea must win, as bar() in judge/judge.py does.

    Parameters
    ----------
    number : float
        The number the idea is judged as: the try count plus one. It is not a whole number
        once old tries have faded.

    Returns
    -------
    float
        The share: 1 - 5%/number, so 0.95 for the first idea and almost 1 for the 300th.
    """
    return 1 - 0.05 / number


def keep_by_days(old, new, number) -> tuple[bool, float]:
    """The old rule: re-deal single days 1,000 times; keep if the candidate wins 95%.

    The bar is the same whatever number the idea is judged as.
    """
    share = float((new[DAY_PICKS].mean(1) < old[DAY_PICKS].mean(1)).mean())
    return share >= 0.95, share


def keep_by_weeks(old, new, number) -> tuple[bool, float]:
    """The judge's rule: re-deal whole weeks; keep if the idea wins bar(number) of the re-deals.

    Parameters
    ----------
    old : np.ndarray
        The champion's daily misses on the validation period.
    new : np.ndarray
        The idea's daily misses on the same days.
    number : float
        The number the idea is judged as: the try count plus one.

    Returns
    -------
    tuple[bool, float]
        Whether the idea is kept, and the share of re-deals it won.
    """
    week = np.minimum(np.arange(len(old)) // 7, 51)  # 52 weeks; the last one is a day or two longer
    sums = np.bincount(week, weights=new - old, minlength=52)
    cnts = np.bincount(week, minlength=52).astype(float)
    share = float((sums[WEEK_PICKS].sum(1) / cnts[WEEK_PICKS].sum(1) < 0).mean())
    return share >= bar(number), share


# Three rules for what a try still counts once the as-of date has advanced.
COUNT_RULES = ["keep the count", "reset the count", "tries fade"]


def try_count(judged, advanced, count_rule) -> float:
    """Count the tries so far, under one of the three rules.

    Parameters
    ----------
    judged : list[int]
        One number per try: how many months the as-of date had advanced when it was judged.
    advanced : int
        How many months the as-of date has advanced by now.
    count_rule : str
        One of COUNT_RULES.

    Returns
    -------
    float
        The try count. The next idea is judged as number try count + 1.
    """
    months_since = [advanced - then for then in judged]
    counts = {
        "keep the count": len(judged),  # a try counts 1 for good
        # only the tries judged under the current as-of date count
        "reset the count": months_since.count(0),
        # a try counts for the share of the months it was judged on that are still in the
        # validation period: the loop's rule, as try_count() in loop.py works it out
        "tries fade": sum(max(0, VALIDATION_MONTHS - months) for months in months_since)
        / VALIDATION_MONTHS,
    }
    return counts[count_rule]


# ---------------- the robot ----------------
# turn 2
TODAY = dict(
    mode="anchored", th=18.0, tc=18.0, win=7, hist=99, shrink=1.0, dow=True, moy=False, junk=()
)
BEST = dict(TODAY, mode="absolute", th=16.0, tc=22.0)  # about as good as this kind of model gets
SETTINGS = ["mode", "th", "tc", "win", "hist", "shrink", "dow", "moy"]


def propose(p, idea, rng, fresh_ids) -> list[dict]:
    """The versions of one idea that the robot tries privately: up to 7.

    Parameters
    ----------
    p : dict
        The champion's settings.
    idea : str
        "new input", or the name of the setting to change (one of SETTINGS).
    rng : numpy.random.Generator
        Where the random tweaks come from.
    fresh_ids : Iterator[int]
        Ids for noise inputs no earlier turn has used.

    Returns
    -------
    list[dict]
        One settings dict per version.
    """
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
        return [
            dict(p, th=round(float(np.clip(p["th"] + rng.normal(0, 2.5), 8, p["tc"])), 1))
            for _ in range(7)
        ]
    if idea == "tc":
        return [
            dict(p, tc=round(float(np.clip(p["tc"] + rng.normal(0, 2.5), p["th"], 30)), 1))
            for _ in range(7)
        ]
    if idea == "shrink":
        return [
            dict(p, shrink=round(float(np.clip(p["shrink"] + rng.normal(0, 0.1), 0.5, 1.5)), 2))
            for _ in range(7)
        ]
    if idea == "win":
        return [
            dict(p, win=int(w))
            for w in rng.choice([w for w in range(3, 29) if w != p["win"]], 7, replace=False)
        ]
    if idea == "hist":
        return [dict(p, hist=h) for h in [6, 9, 12, 18, 99] if h != p["hist"]]
    return [dict(p, **{idea: not p[idea]})]  # dow, moy: flip


def what_changed(old, new) -> dict:
    """Work out what an idea changed in the champion, so the idea can be tried again later.

    Parameters
    ----------
    old : dict
        The champion the idea was tried on.
    new : dict
        The idea: the champion with its change made.

    Returns
    -------
    dict
        The settings the idea set, by name, and under "junk" the made-up inputs it added.
    """
    return {
        **{s: new[s] for s in SETTINGS if new[s] != old[s]},
        "junk": new["junk"][len(old["junk"]) :],
    }


def with_change(p, changed) -> dict:
    """Make an earlier idea's change again, to today's champion.

    Parameters
    ----------
    p : dict
        Today's champion.
    changed : dict
        What the idea changed, as what_changed() gives it.

    Returns
    -------
    dict
        The champion with those settings set, and those made-up inputs added if it lacks them.
    """
    return {
        **p,
        **changed,
        "junk": p["junk"] + tuple(jid for jid in changed["junk"] if jid not in p["junk"]),
    }


class Lab:
    """The toy town, the parallel towns that give the truth, and every score measured so far."""

    def __init__(self, universes=100) -> None:
        # the real toy town: the one a run is judged on, unless judge_on() names another
        self.town = Town(7)
        # The parallel towns, where truth is measured. Each gets a second generator that no other
        # town uses: left to seed + 1, it would be the generator of the next town's first days.
        self.multiverse = [Town(1000 + u, 2000 + u) for u in range(universes)]
        self.cache = {}

    def score(self, p, period, as_of=START_AS_OF) -> np.ndarray:
        """Score a model on one period of the town the run is judged on.

        Each score is worked out once and remembered.

        Parameters
        ----------
        p : dict
            The model.
        period : str
            "practice", "val" or "holdout".
        as_of : pd.Period
            The as-of date the periods are counted back from.

        Returns
        -------
        np.ndarray
            The model's miss on each day of the period.
        """
        key = (tuple(sorted(p.items())), period, as_of)
        if key not in self.cache:
            self.cache[key] = errors(p, self.town, periods(as_of)[period])
        return self.cache[key]

    def truth(self, p, as_of=START_AS_OF) -> float:
        """Measure a model's true error. Each truth is worked out once and remembered.

        Parameters
        ----------
        p : dict
            The model.
        as_of : pd.Period
            The as-of date the periods are counted back from.

        Returns
        -------
        float
            The model's average miss over the validation and holdout periods together, in the
            parallel towns.
        """
        key = (tuple(sorted(p.items())), "truth", as_of)
        if key not in self.cache:
            months = periods(as_of)
            self.cache[key] = float(
                np.mean(
                    [
                        errors(p, t, months["val"] + months["holdout"]).mean()
                        for t in self.multiverse
                    ]
                )
            )
        return self.cache[key]

    def judge_on(self, town) -> None:
        """Judge the runs that follow on another town.

        The scores so far were the last town's, so they are forgotten. The truths stay,
        because the parallel towns stay.

        Parameters
        ----------
        town : Town
            The town whose practice, validation and holdout periods the next runs are scored on.
        """
        self.town = town
        self.cache = {
            (model, what, as_of): score
            for (model, what, as_of), score in self.cache.items()
            if what == "truth"
        }

    def run(
        self,
        start,
        private,
        rule,
        seed,
        new_input_share=0.5,
        turns=300,
        batches=1,
        count_rule="tries fade",
        retry_share=0.0,
        first_id=0,
    ) -> pd.DataFrame:
        """Simulate a run of ideas: one batch, or several batches with a one-month advance between them.

        Parameters
        ----------
        start : dict
            The model the run starts from.
        private : str
            The period the robot tests on before submitting: "val" (the old toy) or "practice".
        rule : Callable
            keep_by_days (the old toy) or keep_by_weeks (the judge's rule).
        seed : int
            The seed of the robot's choices.
        new_input_share : float
            The share of new ideas that add a made-up input. The rest tweak a setting.
        turns : int
            How many ideas the run has.
        batches : int
            The turns are split evenly over this many batches. The as-of date advances one
            month between two batches, so every period moves on a month.
        count_rule : str
            What a try from an earlier batch still counts: one of COUNT_RULES. With one
            batch the three rules agree.
        retry_share : float
            After an advance, the share of turns that are retries: they try again, as it
            was, an idea thrown away under an older as-of date. Such an idea is tried once
            per as-of date.
        first_id : int
            The id of the run's first made-up input. Runs with different ids get different inputs.

        Returns
        -------
        pd.DataFrame
            One row per turn. Every "truth" in it is measured under the run's last as-of
            date, so a run's truths can be compared with each other.
        """
        JUNK.clear()
        # Forget the scores of models with a made-up input (the next run makes its own), and
        # every list of daily misses under a later as-of date (few models come up there twice).
        self.cache = {
            (model, what, as_of): score
            for (model, what, as_of), score in self.cache.items()
            if not dict(model)["junk"] and (what == "truth" or as_of == START_AS_OF)
        }
        rng = np.random.default_rng(seed)
        fresh_ids = itertools.count(first_id)
        last = START_AS_OF + batches - 1  # the as-of date of the last batch
        champ, judged, thrown, rows = start, [], [], []
        for turn in range(1, turns + 1):
            # how many months the as-of date has advanced so far
            advanced = (turn - 1) // (turns // batches)
            as_of = START_AS_OF + advanced

            # the turn's idea: a retry of a thrown-away idea, or a new idea in up to 7 versions
            again = None  # the thrown-away idea this turn tries again, if it is a retry
            # not tried yet under this as-of date
            waiting = [old for old in thrown if old["tried"] < advanced]
            if retry_share and waiting and rng.random() < retry_share:
                idea, again = "retry", waiting[rng.integers(len(waiting))]
                again["tried"] = advanced
                variants = [with_change(champ, again["change"])]
            else:
                if rng.random() < new_input_share:
                    idea = "new input"
                else:
                    idea = str(
                        rng.choice(
                            [
                                s
                                for s in SETTINGS
                                if not (s == "win" and champ["mode"] == "absolute")
                            ]
                        )
                    )
                variants = propose(champ, idea, rng, fresh_ids)

            # the robot tests the versions privately, and submits the best if it beats the champion there
            private_maes = [self.score(v, private, as_of).mean() for v in variants]
            best = variants[int(np.argmin(private_maes))]
            submitted = kept = False
            share = truth_change = val_change = np.nan
            if min(private_maes) < self.score(champ, private, as_of).mean():
                # the judge: the idea is judged as number try count + 1, and its bar follows from that
                submitted = True
                number = try_count(judged, advanced, count_rule) + 1
                judged.append(advanced)
                kept, share = rule(
                    self.score(champ, "val", as_of), self.score(best, "val", as_of), number
                )
                if kept:
                    truth_change = self.truth(best, last) - self.truth(champ, last)
                    val_change = (
                        self.score(best, "val", as_of).mean()
                        - self.score(champ, "val", as_of).mean()
                    )
                    champ = best
                    if again:
                        thrown.remove(again)
                elif not again:
                    thrown.append({"change": what_changed(champ, best), "tried": advanced})
            rows.append(
                dict(
                    turn=turn,
                    idea=idea,
                    submitted=submitted,
                    kept=kept,
                    win_share=share,
                    truth_change=truth_change,
                    val=self.score(champ, "val", as_of).mean(),
                    holdout=self.score(champ, "holdout", as_of).mean(),
                    truth=self.truth(champ, last),
                    inputs=len(champ["junk"]),
                    champion=str(champ),
                    as_of=str(as_of),
                    val_change=val_change,
                    tries=try_count(judged, advanced, count_rule),
                )
            )
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

SEEDS = 30  # in `advance`, each kind of run is simulated this many times, each time on a town of its own
BATCHES = 12  # there, the 300 ideas are spread over this many batches, a one-month advance apart
RETRY_SHARE = (
    0.3  # in the runs with retries: after an advance, 3 turns in 10 try a thrown-away idea again
)
# The runs `advance` compares, each of 300 ideas, and the settings Lab.run() gets for them.
RUNS = {"one batch of 300, no advance": dict()}
RUNS |= {
    f"12 batches of 25, {rule}": dict(batches=BATCHES, count_rule=rule) for rule in COUNT_RULES
}
RUNS |= {
    f"with retries, {rule}": dict(batches=BATCHES, count_rule=rule, retry_share=RETRY_SHARE)
    for rule in COUNT_RULES
}


def compare_systems(lab) -> None:
    """Simulate every batch on the real toy town, under one as-of date, and print what was kept.

    The batches are run for the old toy, each of its two fixes alone, and both together.
    Every turn is saved in loop_sim.csv.

    Parameters
    ----------
    lab : Lab
        The lab the batches are simulated in.
    """
    key_truth = float(np.mean([answer_key_errors(t, VAL + HOLDOUT).mean() for t in lab.multiverse]))
    print(
        f"answer key: validation {answer_key_errors(lab.town, VAL).mean():.3f}  "
        f"holdout {answer_key_errors(lab.town, HOLDOUT).mean():.3f}  truth {key_truth:.3f}"
    )
    batches = []
    for scenario, start in SCENARIOS.items():
        start_val, start_truth = lab.score(start, "val").mean(), lab.truth(start)
        print(
            f"\n{scenario}: start validation {start_val:.3f}  "
            f"holdout {lab.score(start, 'holdout').mean():.3f}  truth {start_truth:.3f}"
        )
        for name, (private, rule) in SYSTEMS.items():
            for seed in range(10):
                df = lab.run(start, private, rule, seed)
                df["scenario"], df["system"], df["seed"] = scenario, name, seed
                df["ledger_gain"] = start_val - df.val
                df["true_gain"] = start_truth - df.truth
                batches.append(df)
        part = pd.concat([n for n in batches if n.scenario.iloc[0] == scenario])
        end, keeps = part[part.turn == 300], part[part.kept]
        print(
            pd.DataFrame(
                {
                    "kept": keeps.groupby("system").size().reindex(list(SYSTEMS), fill_value=0)
                    / 10,
                    "noise inputs kept": end.groupby("system").inputs.mean(),
                    "flukes kept": keeps[keeps.truth_change >= 0]
                    .groupby("system")
                    .size()
                    .reindex(list(SYSTEMS), fill_value=0)
                    / 10,
                    "ledger gain": end.groupby("system").ledger_gain.mean(),
                    "true gain": end.groupby("system").true_gain.mean(),
                    "final holdout": end.groupby("system").holdout.mean(),
                }
            )
            .reindex(list(SYSTEMS))
            .round(3)
            .to_string()
        )
    pd.concat(batches).to_csv("loop_sim.csv", index=False)
    print(
        "\nAverages over 10 batches of 300 turns each. MWh per day; gain = how much the average miss shrank."
    )
    print("'flukes' = kept changes that made the true error worse.")


def compare_counts(lab) -> None:
    """Spend the same 300 ideas in one batch and in 12 batches with advances, and print what was kept.

    The 12 batches are run under the three rules for the try count, without and with retries.
    Both of the robot's fixes are on. Every turn is saved in loop_sim_advance.csv.

    Parameters
    ----------
    lab : Lab
        The lab the runs are simulated in.
    """
    runs = []
    for scenario, start in SCENARIOS.items():
        summary = []
        for seed in range(SEEDS):
            # Every seed gets a town of its own to be judged on, and made-up inputs of its own,
            # so the seeds are separate experiments: none shares another's luck.
            lab.judge_on(Town(4000 + seed, 5000 + seed))
            for name, settings in RUNS.items():
                df = lab.run(
                    start, "practice", keep_by_weeks, seed, first_id=10_000 * seed, **settings
                )
                df["scenario"], df["run"], df["seed"] = scenario, name, seed
                runs.append(df)
                keeps = df[df.kept]
                summary.append(
                    {
                        "run": name,
                        "judged": df.submitted.sum(),
                        "kept": len(keeps),
                        "lucky keeps": (keeps.truth_change >= 0).sum(),
                        "ledger gain": -keeps.val_change.sum(),
                        "true gain": -keeps.truth_change.sum(),
                        "bar at the end": 100 * bar(df.tries.iloc[-1] + 1),
                    }
                )
        print(f"\n{scenario}")
        print(pd.DataFrame(summary).groupby("run", sort=False).mean().round(3).to_string())
    pd.concat(runs).to_csv("loop_sim_advance.csv", index=False)
    print(f"\nAverages over {SEEDS} runs of 300 ideas each, every run judged on a town of its own.")
    print("MWh per day; gain = how much the average miss shrank.")
    print(
        "'judged' = ideas that reached the judge. 'lucky keeps' = kept changes that did not make the true error smaller."
    )
    print(
        "From the best model, read 'kept': it is taken to be as good as this kind of model gets, so every keep is a lucky one."
    )
    print(
        "'ledger gain' = the gains of the kept ideas on the validation period they were judged on, added up."
    )
    print(
        "'true gain' = the same ideas' gains in the 100 parallel towns, on the months of the run's last as-of date."
    )
    print("'bar at the end' = the share of reshuffles (%) the 301st idea would have to win.")


if __name__ == "__main__":
    if not sys.argv[1:]:
        compare_systems(Lab())
    elif sys.argv[1:] == ["advance"]:
        compare_counts(Lab())
    else:
        sys.exit("Run it with no argument, or with `advance`: see the top of this file.")
