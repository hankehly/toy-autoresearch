# Toy autoresearch: a fake town's electricity use

The smallest working version of an agent research loop, with guards against overfitting.

    lab/        the agent's files: program.md (your brief; the loop fills in its dates each
                turn), model.py, check.py, practice.py (scores a change on the practice
                period), ledger.tsv (the agent's copy of the ledger). Its data is not a
                file here: see data/
    judge/      the frozen judge. It works out the four periods from the as-of date
    as_of.txt   the as-of date: one month, the last one the research may use. Only
                `loop.py advance` and `loop.py reset` write it
    store.py    the data store's functions: list months, read months, write a month.
                The loop, the judge and the scripts reach the data only through them
    data/       not in git; `uv run scripts/make_data.py` builds it:
                store/   the data store: one folder per month, month=2023-01/part.parquet
                         (date, temp_forecast, load). Each sandbox is shown some of these
                         folders, read-only, and those are the only months it can see
                future/  a pretend future, 2023-01 to 2042-12, that only ingest.py reads
    loop.py     each turn: fresh workspace -> fresh agent makes one change -> the loop
                measures it on the practice period -> judge compares -> keep or throw away -> ledgers
    ledger.tsv  the full ledger, validation scores included; the agent never sees it
    history/    every model the agent wrote, plus Claude Code's whole response for the turn
                (the reply and the cost, or the errors if the run failed)
    docker/     the two sandboxes and the proxy (Docker images), see "Isolation" below:
                model-sandbox  where judge.py runs model.py: no internet, and nothing from your
                               Mac but model.py and the months before the ones being scored
                agent-sandbox  where the agent works: it sees only the turn's workspace and
                               the months up to the end of the practice period
                agent-proxy    the agent sandbox's only way out: it lets through Claude's API alone
    scripts/    helpers outside the main loop:
                make_data.py       rebuilds the fake town (seeded, so you get the same data)
                ingest.py          lets time pass: moves months from the pretend future into the store
                answer_key.py      the recipe that made the data, as a model: about the best possible score
                night_sim.py       simulates whole nights of 300 ideas, in one night or over 12 nights
                                   with advances, and checks every keep against the truth
                bundle_sim.py      tests whether small changes add up when submitted together
    tests/      the tests (pytest). `uv run pytest` runs them all, in about 3.5 minutes:
                conftest.py        what the test files share: the Docker images, and for each test
                                   file a temporary data folder and as-of file, so data/ and
                                   as_of.txt are never touched
                test_sandboxes.py  proves the isolation holds (about 30 seconds)
                test_loop.py       proves each kind of turn is logged as promised, and that old tries
                                   count for less after an advance (about 1.5 minutes)
                test_as_of.py      proves the periods follow the as-of date, that an advance
                                   moves them, and that no date is written into the code (about 1 minute)
                test_store.py      proves the store holds the same days as ever, and that only
                                   ingest.py adds months (about 30 seconds)
                test_size.py       scores a 20-year, 303-column store (about 30 seconds)
    .github/    GitHub runs `uv run pytest` on every pull request and every push to main
    docs/       the diagrams below, and adr/: why the periods stay fixed between advances,
                and why old tries count for less after one

## One turn of the loop

![One turn: lab/ is copied to a throwaway workspace, the agent changes model.py and replies, the loop measures the change on the practice year, a submitted change goes to the judge, which keeps it or throws it away, every outcome is written to the full ledger, and the next turn's agent reads a copy without the validation columns](docs/research-loop.drawio.png)

The diagram is a `.drawio.png`: it displays as an image and also opens for editing in draw.io.

## Who sees which months

No period is written into the code as dates. One file, `as_of.txt`, holds one month: the
as-of date, the last month the research may use. The judge counts the four periods back
from it (`uv run judge/judge.py periods` prints them):

    period      rule                                    as-of date 2025-12   as-of date 2026-06
    holdout     the 6 months ending at the as-of date   2025-07 to 2025-12   2026-01 to 2026-06
    validation  the 12 months before that               2024-07 to 2025-06   2025-01 to 2025-12
    practice    the 12 months before that               2023-07 to 2024-06   2024-01 to 2024-12
    training    every earlier month in the store        2023-01 to 2023-06   2023-01 to 2023-12

The agent has the training and practice periods. It never sees the validation period, where
the judge decides what is kept, and the holdout period is scored once, at the end. The
three lengths are settings at the top of `judge/judge.py`.

Every score walks forward a month at a time: the model is trained on everything before
the month, then predicts each day of it. So with the as-of date 2025-12, 2024-07 is
predicted from 18 months of history and 2025-06 from 29.

Real data keeps arriving, so the toy has a pretend clock: `uv run scripts/ingest.py --through 2026-06`
moves the months up to 2026-06 from `data/future/` into the store, whole months only, and
never changes a month already there. In a real project the job that builds the data does this.

Months that arrive move nothing by themselves. The periods stay where they are, and no
sandbox is shown a month after the as-of date, until a person runs `uv run loop.py advance`
between nights. That moves the as-of date to the store's newest month, so all four periods
move forward by the same number of months. It then measures the champion on the new practice
and validation periods and adds one `advanced` row to the ledgers. It changes no model and
scores nothing on the holdout period.

The periods must not move while ideas are being judged: the keep rule counts tries against
one validation period and compares each idea with a champion measured on it
(`docs/adr/0001-periods-frozen-at-an-as-of-date.md`). Scores from before and after an
advance come from different months, so every ledger row records its as-of date.

![The same rule at five moments: months that arrive move no period, and advances of 6, 1 and 5 months move all four periods forward while the training period grows](docs/periods.drawio.png)

## Run it

Needs Docker Desktop (running), Claude Code and [uv](https://docs.astral.sh/uv/).

    uv sync                           # once: Python 3.12 and the locked packages, into .venv
    uv run scripts/make_data.py       # once: build data/ (the same days every time; it is not in git)
    uv run loop.py build              # build the Docker images (run, advance and holdout also do this first)
    uv run pytest                     # run every test (about 3.5 minutes; no login token, and data/ is not touched)
    uv run pytest -k proxy            # or only the tests with "proxy" in their names
    claude setup-token                # once: a login token the agent sandbox can use
    export CLAUDE_CODE_OAUTH_TOKEN=<the token it prints>
    uv run loop.py run 5              # five agent turns
    uv run loop.py holdout            # score the holdout period once, at the end
    uv run scripts/ingest.py --through 2026-06   # later: new months arrive in the store
    uv run loop.py advance            # by hand, between nights: move the as-of date to the store's newest month
    uv run loop.py reset              # start over from the baseline, at the as-of date the research started under
    uv run scripts/night_sim.py       # the overnight experiment, no agent needed (about 5 minutes)
    uv run scripts/night_sim.py advance   # the same 300 ideas over 12 nights with advances (about 16 minutes)

The ledgers and history/ in this folder are from the example run below (agent: Claude Sonnet
via `claude -p`; change the model in loop.py). `uv run loop.py reset` clears them.

## The example run (average miss per day, MWh; lower is better)

                                                  practice year      validation year
    turn  idea                                    miss   won     miss   won    needed  verdict              holdout
    0     average of the last 7 days              5.66   -       5.78   -      -       start                5.82
    1     weekday + heating/cooling regression    3.04   100%    2.92   100%   95%     KEPT                 2.75
    2     model log(load) instead of load         3.03   66%     2.94   10%    97.5%   thrown away
    3     add yesterday's temperatures            2.88   99.9%   2.89   68%    98.3%   thrown away
    4     move thresholds to 18 / 22 C            3.01   66%     2.98   9%     98.8%   thrown away
    5     correct with yesterday's error          3.04   19%     2.91   62%    99%     thrown away
    6     add yearly sine/cosine terms            3.09   10%     2.92   29%    99.2%   thrown away
    7     fit by least absolute deviations        -      -       -      -      -       dropped on practice
    8     add squared heating/cooling degrees     3.21   2%      -      -      -       dropped on practice
          answer key (not part of the run)                       2.83                                       2.77

Every row was measured under the as-of date 2025-12: the practice period is 2023-07 to
2024-06, the validation period 2024-07 to 2025-06 and the holdout period 2025-07 to 2025-12.
"Won" is the share of 10,000 reshuffles of the year's weeks in which the idea beat the
champion of the time. The practice columns were measured afterwards, for turns 0-6.
The answer key is the recipe that made the data, which a real project doesn't have. The
loop never runs it; `scripts/night_sim.py` prints its scores, and
`uv run judge/judge.py score val scripts/answer_key.py` gives the same.

Checked afterwards on 50 parallel towns: turns 2, 4 and 5 were truly no better, so the
judge was right. Turn 3 was truly better by 0.04 MWh, but one scored year was too small
to prove it, so both the old and the new rule threw it out. Small exams miss small gains.
On the practice year, turn 3 is the only one of turns 2-6 to win more than 90% of the
reshuffles. An idea is judged only at 90% or more: `SUBMIT_PCT` in loop.py, which the loop
enforces with its own practice measurement and writes on top of the agent's ledger. Winning
90% of reshuffles means a gain of about 1.3 times its own luck, whatever the units, so the
setting carries over to other problems. (Turns 2-6 were judged before this rule existed.)
Turns 6 and 7 came later, inside the sandboxes (see "Isolation"). Turn 7 found its idea
worse on the practice year (3.19 against 3.04, by its own scorer) and dropped it, so it
didn't count as a try. It also copied the champion back, so that idea's code is lost.
Turn 8 was the first with practice.py and the structured reply: the loop measured its idea
at 2% of practice reshuffles, the same numbers the agent reported, and it wasn't submitted.

## Two ledgers

Every turn adds a row to the full ledger, `ledger.tsv`, which stays on the host. The agent's
copy, `lab/ledger.tsv`, is rewritten from it after every turn without the validation columns:

    full ledger    turn, as_of, idea, practice_champion, practice, practice_wins,
                   val_mae, win_pct, bar_pct, verdict, lesson, reply_check
    agent's copy   turn, as_of, idea, practice_champion, practice, practice_wins, verdict, lesson

The agent sees practice numbers, which come from months it already has, and the judge's
verdict. If it also saw validation scores, it could pick and combine near-misses by them,
which is tuning on the judge's period one row at a time (finding 6 in CLAUDE.md).

`as_of` is the as-of date the row was measured under. Old rows are never measured again, so
scores from rows with different as-of dates come from different months and can't be compared
directly. The line on top of the agent's copy always gives the champion's newest practice
score: after an advance, the one in the `advanced` row.

The loop measures the practice numbers itself, with the model in the model sandbox, so
they don't depend on what the agent reports. The agent finishes each turn with a structured
reply (its idea, a lesson, the numbers `practice.py` printed, and whether to submit), which
the loop checks against a JSON schema. `reply_check` records whether the agent's practice
numbers match the loop's: `ok`, `practice mismatch`, `not run` (the agent reported none), or
`-` when the loop had nothing to compare them with.

    verdict              when                                                        a try?
    KEPT                 judged, and cleared the bar                                 yes
    thrown away          judged, and didn't clear it                                 yes
    dropped on practice  not submitted, or won under SUBMIT_PCT of practice          no
    no change            model.py is identical to the champion                       no
    crashed              model.py failed in the loop's practice run or the judge's   no
    invalid reply        no valid structured reply; practice is still measured       no
    start                the baseline row                                            no
    advanced             the as-of date moved; the champion was measured again       no

Only a try raises the bar. An `advanced` row is not a turn either, so it has no turn number.
The second line on top of the agent's copy gives the try count and the bar the next idea
must clear (see "The keep rule").

## What night_sim.py found (10 nights of 300 ideas each, MWh)

"Flukes" are kept changes that made the true error worse. "Ledger" is how much the
scored-year miss shrank; "truth" is how much the true miss shrank (100 parallel towns).

                       starting from turn 2's model     starting from the best model
                       flukes   ledger   truth           flukes   ledger   truth
    old toy            9.3      +0.45    +0.22           12.2     +0.21    -0.03
    hide answers only  0.7      +0.30    +0.23           2.0      +0.07    -0.01
    rising bar only    0.6      +0.22    +0.17           0        0        0
    both (this toy)    0        +0.24    +0.20           0        0        0

## What `night_sim.py advance` found (30 runs of 300 ideas each, MWh)

After an advance, some of the months an old try was judged on have left the validation
period. What should that try still count? The simulation spends the same 300 ideas in one
night, and in 12 nights of 25 with a one-month advance between them, under three rules:
keep the count, reset it at each advance, or let old tries fade (this toy's rule, see
"The keep rule"). "Lucky" is the number of lucky keeps per run. From turn 2's model, a
lucky keep is a kept change that did not make the true error smaller. The best model is
taken to be as good as this kind of model gets, so there every kept change counts as one.
Every run is judged on a town of its own.

                                           starting from turn 2's model     starting from the best model
                                           lucky   ledger   truth           lucky   ledger   truth
    one night of 300, no advance           0.07    +0.15    +0.13           0.10    0        0
    12 nights, keep the count              0.10    +0.22    +0.18           0.03    0        0
    12 nights, reset the count             0.60    +0.25    +0.21           0.60    +0.01    0
    12 nights, tries fade (this toy)       0.10    +0.22    +0.18           0.03    0        0
    with retries, keep the count           0.07    +0.22    +0.18           0.07    0        0
    with retries, reset the count          0.43    +0.25    +0.22           0.53    +0.01    0
    with retries, tries fade (this toy)    0.10    +0.21    +0.18           0.07    0        0

"With retries": after each advance, 3 turns in 10 try again, as it was, an idea that was
thrown away under an older as-of date. The brief allows this, and a retry counts as a try.

- Letting old tries fade kept about one lucky idea in ten runs, far under one per 300
  ideas, and about as many as never forgetting a try: one more, in the 120 runs of the two
  rules. It was an idea that won 99.94% of reshuffles, enough as number 74 (bar 99.93%)
  and not as number 103 (bar 99.95%).
- Resetting the count kept one lucky idea in about every two runs.
- A retry can be a lucky keep, like any other try, and is about as likely to be one as a
  new idea. Over 200 runs of the two fading rows (`SEEDS = 200`), the rule kept 14 lucky
  ideas with retries and 15 without from turn 2's model, and 27 and 21 from the best
  model: 0.03 more per run there, with luck of 0.02. Retries were 27% of the turns and 10
  of the 41 lucky keeps.
- In the first year, fading and keeping the count are nearly the same rule, because new
  tries arrive faster than old ones fade: after the 300 ideas the bar stood at 99.95%
  against 99.98%. They part later. At a steady pace the fading count levels off (here near
  100 tries) while a kept count grows for good, and when the research pauses the fading
  bar comes back down.
- The same 300 ideas found more real gain over 12 nights than in one (+0.18 against +0.13).
  In one night every idea meets the same validation months, and in 9 of the 30 single
  nights nothing was kept at all. With advances that happened once. (Truth is measured on
  the months of each run's last as-of date, so on later months for the 12 nights. That
  moves it little: the ideas kept on the first night, the same in both, measure +0.087 on
  the earlier months and +0.089 on the later ones.)

## The keep rule

The judge re-deals the 52 scored weeks 10,000 times (some weeks twice, some not at all)
and keeps a change only if it has the lower average miss in nearly all of the re-deals.
The bar rises with the try count: the next idea must win 1 - 5% / (try count + 1) of the
re-deals. So the 1st idea must win 95%, the 10th 99.5%, the 100th 99.95%.
Added up over a whole night, that keeps the expected number of lucky keeps below one.
Whole weeks are re-dealt together because neighbouring days share weather.

A try counts 1 until the next advance. An advance replaces some of the validation months
the try was judged on, and from then on the try counts for the share of them that is left
(`docs/adr/0002-old-tries-fade-after-an-advance.md`). In plain words: an old try is worth
less the more the validation period has moved on since. With 8 tries in the ledger:

    months advanced since the 8 tries   each try counts   try count   bar for the next idea
    0                                   12/12             8.00        99.44%
    1                                   11/12             7.33        99.40%
    6                                   6/12              4.00        99.00%
    12 or more                          0                 0.00        95.00%

It is like a reference letter that mentions twelve projects. Each month one project is
swapped for a new one, so the letter says a little less about the current work, and after
a year it says nothing. The second line of the agent's ledger gives the try count and the
bar that follows from it.

## Three guards against overfitting

1. The agent never has the answers for the period it's graded on, so it can't tune on it.
2. The bar rises with the try count, so a long night can't win by sheer volume. A try
   counts for less only once an advance has replaced months it was judged on.
3. The holdout is scored once. If you change things because of what it says, it becomes
   another validation set, so then you need fresh months to check on.

## Isolation

![The agent sandbox, the host and the model sandbox side by side, with the only things that pass between them: the lab files and the agent's months of data in, model.py out, model.py with the earlier months and the messages in, one guess per day out](docs/sandboxes.drawio.png)

Isolation, not instructions, keeps the agent's code away from the answers:

- **The host** (your Mac) runs only code a human wrote, `loop.py` and `judge.py`, and holds
  the whole data store, the holdout included.
- **The agent sandbox** is a container that sees only the turn's workspace and, read-only at
  `/data`, the month folders up to the end of the practice period. Its network has one way
  out, the proxy, which lets through Claude's API and refuses everything else. The agent's
  web tools are off, and only a plain `model.py` comes back from the sandbox.
- **The model sandbox** is where the judge runs `model.py`: a container with no internet, no
  writable files, and nothing from your Mac except `model.py` itself and the month folders
  before the first scored month, all read-only. The judge sends it messages, one line of
  JSON at a time, and reveals each day's load only after the model has guessed it.

The months a sandbox can see are exactly the month folders mounted into it. A month that
isn't mounted isn't there to be read, whatever the code inside tries.

The networks behind this: a container can reach only what's on the networks it's plugged
into. The agent sandbox is plugged into one internal network, whose only other member is the
proxy. The proxy is also plugged into Docker's default network, which leads to the internet.
`HTTPS_PROXY` tells Claude Code to send its requests to the proxy; anything that ignores it
finds no route out.

![The agent sandbox sits on an internal network with the proxy; the proxy is also on Docker's bridge network and lets through only api.anthropic.com; the model sandbox has no network and talks to the judge through stdin and stdout](docs/networks.drawio.png)

Before the sandboxes, the judge ran `model.py` in its own process, so a 6-line model could
read today's answer from the judge's variables. It was kept, and the holdout agreed with it.
`uv run pytest tests/test_sandboxes.py` now runs five such cheats and checks that each one crashes.

The proxy stays up after a run. Its log lists every request the agent sent through it, until
the next run or test run starts a fresh proxy. Attempts to go around the proxy simply fail,
and leave no trace there.

    docker exec toy-agent-proxy cat /var/log/squid/access.log

What the sandboxes don't fix: Docker is a strong fence for an over-eager agent, not a
guarantee against hostile code. And each verdict still tells the agent one thing about the
judge's period: whether that idea cleared the bar.
