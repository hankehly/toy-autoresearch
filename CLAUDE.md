# Toy autoresearch

Context for Claude Code. This project started in a Claude chat (2026-11-05 to 2026-11-08); this file
carries over what was decided and learned there. README.md describes the files and commands.

## Goal
Hank wants AI agents to run the iterative research and model-building for predictive
problems that use traditional ML, such as next-day electricity load and horse racing. This
folder is the smallest working version: a made-up town's daily electricity use, an agent that
improves `lab/model.py` one idea per turn, and a frozen judge that decides what's kept.
The lessons are meant to carry over to real problems like these.

## How it works
- `loop.py run N`: each turn copies `lab/` to a temp folder, where a fresh `claude -p` agent
  reads `program.md` and `ledger.tsv`, changes only `model.py`, checks it with `check.py` and
  scores it against the champion on the practice period with `practice.py`. It finishes with a
  structured reply (idea, lesson, practice numbers, submit or not), which the loop checks
  against `REPLY_SCHEMA` in `loop.py`. Only `model.py` comes back. The loop measures it on
  the practice period itself (`judge.py practice`, in the model sandbox). If the agent
  submitted it and the loop measured at least `SUBMIT_PCT` (90%) of practice reshuffles won,
  `judge/judge.py` compares it with the champion on the validation period, and it is kept or
  thrown away.
- `lab/program.md` is a template: it names no date. Each turn `brief()` in `loop.py` fills
  in the `{names}` (the agent's months, the practice period, the period lengths) and writes
  the result into the workspace. It uses Python's `str.format`, so a brace meant to reach
  the agent as text must be doubled (`{{`), and a misspelt name stops the turn.
- Two ledgers. `ledger.tsv` at the repo root is the full record, validation scores
  included, and the try count behind the bar comes from its verdicts and its `as_of`
  column. `lab/ledger.tsv` is the agent's
  copy, rewritten from it after every turn: practice numbers, verdict and lesson, and no
  number from the validation period. Verdicts: `KEPT` and `thrown away` were judged and count
  as tries; `dropped on practice` (changed, but not submitted or under `SUBMIT_PCT`),
  `no change`, `crashed` and `invalid reply` weren't and don't. `reply_check` in the full ledger says whether the
  agent's practice numbers matched the loop's. Every row records the as-of date it was
  measured under (`as_of`); old rows are never measured again. An `advanced` row is added by
  `loop.py advance`: it has no turn number, is not a try, and holds the champion's practice
  and validation scores under the new as-of date. The top line of the agent's copy uses the
  champion's newest practice score, and the second gives the try count (to two decimals)
  with the bar the next idea must clear.
- Data store (`store.py`; `data/`, not in git). One folder per month in the layout Spark
  writes: `data/store/month=2023-01/part.parquet`, columns `date`, `temp_forecast`, `load`.
  The loop, the judge and the scripts reach the data only through `store.py`'s functions
  (list months, read months, write a month), so a real store later changes that one file.
  `scripts/make_data.py` builds `data/future/` (a pretend future, 2023-01 to 2042-12) and
  fills the store through 2025-12. `scripts/ingest.py --through 2026-06` moves later months
  from the future into the store: whole months, never changing one already there. Nothing
  reads a month after the as-of date, and no sandbox is shown one.
- Isolation (`docker/`). The host runs only code a human wrote (`loop.py`, `judge.py`) and
  holds the whole data store. The agent works in the agent sandbox, a container that sees
  only the turn's temp folder and, read-only at `/data`, the month folders up to the end of
  the practice period; its network's only way out is the proxy (squid), which lets through
  api.anthropic.com alone. The judge runs `model.py` in the model sandbox (no network,
  read-only, nothing but `model.py` and the month folders before the first scored month).
  The runner loads those months when it starts, and the judge talks to it in one-line JSON
  messages (fit, predict, reveal): `predict` carries every column of today except `load`,
  and `reveal` hands over each day's load only after the guess. The months a sandbox can see
  are exactly the month folders mounted into it. `tests/test_sandboxes.py` proves the
  isolation holds.
- Periods and the as-of date (issue #4; words in `GLOSSARY.md`, reasons in `docs/adr/0001`).
  Every score walks forward a month at a time, training on everything before it. No period
  is written down as dates: `as_of.txt` in the repo root holds one month, the as-of date, and
  `periods()` in `judge/judge.py` counts the periods back from it with three settings there.
  `uv run judge/judge.py periods` prints them; `loop.py` asks the judge the same way.

  | Period | Rule | As-of date 2025-12 (a fresh checkout) | As-of date 2026-06 |
  |---|---|---|---|
  | Holdout, scored once at the end | The 6 months ending at the as-of date | 2025-07 to 2025-12 | 2026-01 to 2026-06 |
  | Validation, the agent never has it | The 12 months before that | 2024-07 to 2025-06 | 2025-01 to 2025-12 |
  | Practice, the agent's own | The 12 months before that | 2023-07 to 2024-06 | 2024-01 to 2024-12 |
  | Training | Every earlier month in the store | 2023-01 to 2023-06 | 2023-01 to 2023-12 |

  The judge stops if the store does not reach the as-of date or has no month before the
  practice period.
- `loop.py advance`, by hand, between nights: moves the as-of date to the store's newest
  month (one month or many), measures the champion on the new practice and validation
  periods, and adds the `advanced` row. It changes no model and scores nothing on the
  holdout period. The tries from before it then count for less (the try count, below), so
  the bar eases a little. An advance happens whole or not at all: if it is cut short (the
  champion can't be measured, a timeout, Ctrl-C), the as-of date and both ledgers are put back. With
  nothing newer in the store it stops. `loop.py reset` puts the as-of date back to the one
  in the ledger's `start` row and leaves the store alone. `loop.py holdout` prints each
  model's validation score measured now, beside its holdout score.
- Keep rule: reshuffle the 52 scored weeks 10,000 times. An idea must win at least
  1 - 5% / (try count + 1) of reshuffles (95% for the 1st idea, 99.5% for the 10th, 99.95%
  for the 100th), which keeps the expected number of lucky keeps under one for a 300-idea
  night.
- Try count (issue #5; `try_count()` in `loop.py`; reasons in `docs/adr/0002`). A try counts
  in proportion to how many of the months it was judged on are still in the validation
  period: for each judged row, `(12 - months between its as-of date and the current one) / 12`,
  never below zero, with the judge's validation length for the 12. In plain words: an old try is
  worth less the more the validation period has moved on since. It is a whole number until
  an advance follows a try, so without an advance every bar is what it was before this rule.

  | Months advanced since 8 tries | Each try counts | Try count | Bar for the next idea |
  |---|---|---|---|
  | 0 | 12/12 | 8.00 | 99.44% |
  | 1 | 11/12 | 7.33 | 99.40% |
  | 6 | 6/12 | 4.00 | 99.00% |
  | 12 or more | 0 | 0.00 | 95.00% |

  The judge's `bar` and `compare` commands take the idea's number (try count plus one),
  which need not be whole.
- `scripts/answer_key.py` is the recipe that made the data (validation 2.83, holdout 2.77 MWh). It is
  close to the best possible but not a hard floor: adding yesterday's temperatures beats it
  slightly, because that sharpens the estimate of today's actual temperature. A real project
  has no answer key, so the loop never runs it; score it with
  `uv run judge/judge.py score val scripts/answer_key.py`.
- `scripts/night_sim.py` re-implements the judge for speed (it reproduces the judge's scores
  exactly, under every as-of date from 2025-12 to 2026-12) and simulates whole nights with a
  random-tweak robot. Its towns are made by `make_data.py`'s own `town()`: generators 7 and
  8 give the real town, pretend future included. True error is measured on 100 parallel
  towns: the same recipe with fresh weather and noise. `uv run scripts/night_sim.py advance`
  spends the 300 ideas over 12 nights with a one-month advance between them, under three
  rules for the try count (finding 8). If you change the judge's rules or the loop's
  `try_count()`, change `scripts/night_sim.py` to match.
- `scripts/bundle_sim.py` tests whether small changes add up (finding 6).

## What we learned (MWh per day; "luck" = one standard deviation of a measured gain)
1. Run 1, old rules (the agent had the scored year; a fixed 95% bar on single days): it found
   weekday effects and temperature. In one turn the agent tried 7 thresholds on the scored
   year and submitted the best; it looked better there and worse on the holdout, and the keep
   rule threw it out.
2. Simulated 300-idea nights (half setting tweaks, half pure-noise inputs), old rules: about
   9 flukes kept per night; the ledger claimed +0.45 while the truth was +0.22, and real
   progress stopped after about 30 ideas. From an already-optimal model: 12 noise inputs
   kept, ledger +0.21, truth -0.03. Hiding the scored year cut flukes to under 1 per night;
   adding the rising bar made it 0 (ledger +0.24, truth +0.20), at a small cost in real gains.
3. Neighbouring days are only weakly linked in this toy (about +0.1), so reshuffling whole
   weeks barely matters here. It is kept for real data (weather spells, same-day race conditions).
4. Run 2, new rules (the current ledger and history): turn 1 went straight to a weekday plus
   heating/cooling regression (validation 2.92, holdout 2.75), and the next 4 ideas were rejected.
   Checked on 50 parallel towns, 3 were truly no better, but turn 3 (add yesterday's
   temperatures) was truly better by 0.039, while one scored year's luck was 0.049.
   Small exams miss small gains.
5. Luck shrinks with the square root of the size of the scored set. A pure-noise input
   (best of 7) clears the old rule 11% / 5% / 0% of the time with 1 / 10 / 100 scored years.
   That is why Karpathy's autoresearch (about 21 million scored tokens) gets away without these
   guards, and why a year of races (about 3,450 JRA races; ROI luck of about 5 points at odds
   of 10) needs them more.
6. Bundles: ten real small changes (each 0.018 against luck of 0.021; clears the bar 20% alone)
   add up to 0.181 against 0.066 (clears it 89%), because gains add while luck grows only with
   the square root. Ten noise near-misses picked on the scored year, bundled, cleared the bar
   45% of the time while truly 0.025 worse; picked on the practice year, 0%.
   Pick the pieces on data the final test doesn't use.
7. Before the sandboxes, the judge ran `model.py` in its own process, next to the answers. A
   6-line model that read today's load from the judge's loop variable was kept as the 6th
   idea (won 100%; validation 2.14 and holdout 2.06 with noise added, 0.00 without), and it passed
   `check.py`. The holdout agreed, so every statistical guard failed at once. Isolation, not
   statistics, stops this; the model sandbox now makes it crash (`tests/test_sandboxes.py`).
8. What a try counts after an advance (`scripts/night_sim.py advance`: the same 300 ideas in
   one night, and in 12 nights of 25 with a one-month advance between them; 30 runs, each
   judged on a town of its own). Each cell: from turn 2's model / from an already-optimal
   one. From turn 2's model a lucky keep is a kept change that did not make the true error
   smaller. From the already-optimal model every kept change counts as one.

   | 300 ideas | Lucky keeps per run | The same with retries | Ledger gain | True gain | Bar after the 300 ideas |
   |---|---|---|---|---|---|
   | One night, no advance | 0.07 / 0.10 | | +0.15 / 0 | +0.13 / 0 | 99.97% |
   | 12 nights, keep the count | 0.10 / 0.03 | 0.07 / 0.07 | +0.22 / 0 | +0.18 / 0 | 99.98% |
   | 12 nights, reset the count | 0.60 / 0.60 | 0.43 / 0.53 | +0.25 / +0.01 | +0.21 / 0 | 99.70% |
   | 12 nights, tries fade (the loop's rule) | 0.10 / 0.03 | 0.10 / 0.07 | +0.22 / 0 | +0.18 / 0 | 99.95% |

   Gains and bars are without retries. A retry tries again, as it was, an idea thrown away
   under an older as-of date: 3 turns in 10 after each advance, the idea picked at random.
   - Fading kept about one lucky idea in ten runs: far under one per 300 ideas, and about as
     many as never forgetting a try. It kept one more in the 120 runs of each: an idea that
     won 99.94% of reshuffles, enough as number 74 (bar 99.93%) and not as number 103 (99.95%).
   - Resetting kept one lucky idea in about every two runs, several times as many, for a
     little more real gain (+0.03; the gap's luck is 0.014).
   - A retry can be a lucky keep like any other try, and is about as likely to be one as a
     new idea. Over 200 runs of the two fading rows (`SEEDS = 200`): 14 lucky keeps with
     retries and 15 without from turn 2's model, 27 and 21 from the already-optimal one
     (0.03 more per run, a gap whose luck is 0.02). Retries were 27% of the turns and 10 of
     the 41 lucky keeps. One of the 10 was judged three times: it won 80%, then 99.44%, then
     99.92% against a bar of 99.89%, as the validation period moved on a month each time.
     For that idea, a retry after a one-month advance was far from the same test again.
   - Why: each judged idea may be a lucky keep with a chance of at most 5% / (try count + 1).
     With about 17 ideas judged a night, those chances add up over the 12 nights to 0.30
     with a kept count, 0.33 with fading and 2.1 with a reset. The measured counts are a
     third of that or less, in the same order, as expected when a made-up input truly hurts
     a little.
   - In the first year fading and keeping the count are nearly the same rule, because new
     tries arrive faster than old ones fade. They part later: at this pace the fading count
     levels off near 100 (a bar of 99.95%, and by the same sum at most 0.1 lucky keeps a
     year), a kept count grows for good, and when the research pauses the fading bar comes
     back down. The gap is wider at a slower pace: with 3 ideas judged a night, the fading
     count levels off at 20 (99.76%), where a kept count reaches 108 in three years (99.95%).
   - The same 300 ideas found more real gain over 12 nights than in one (+0.18 against
     +0.13; the gap's luck is 0.017). In one night every idea meets the same validation
     months: in 9 of the 30 single nights nothing was kept, against 1 with advances. (Truth
     is measured on the months of each run's last as-of date, so on later months for the 12
     nights. That moves it little: the ideas kept on the first night, the same in both,
     measure +0.087 on the earlier months and +0.089 on the later ones.)
   - Two checks were made once, with throwaway scripts that are not in this repo (the pull
     request for issue #5 has the numbers). Judged on the real town alone, the lucky keeps
     were alike (fading 0 to 0.07, keeping 0 to 0.07, resetting 0.27 to 0.70), but the single
     night gained more than the 12 nights (+0.19 against +0.14): one town's validation
     months are one draw of luck, which is why each run now gets its own town. And when the
     robot retried the same few ideas after every advance, up to 12 times in the year,
     fading kept 0.07 / 0.10 lucky ideas per run.

## Open threads
- Allow practice-year bundles: let the agent submit several ideas that each helped on its
  practice year as one try, and see whether it captures turn 3's real gain.
- The agent's ledger now shows practice numbers and verdicts only (issue #1). Unproven so
  far: run a real night and see whether practice numbers steer it. What still leaks is one
  fact per judged idea: whether it cleared the bar.
- An idea is judged only if it wins at least 90% of practice reshuffles: `SUBMIT_PCT` in
  `loop.py`, enforced by the loop and written on top of the agent's ledger, where step 5 of
  program.md points. 90% means a gain of about 1.3 times its own luck, in any units. It is
  a starting point (in run 2 only turn 3 would have cleared it). Revisit it once the full
  ledger has a few dozen judged ideas.
- Near-misses: the brief (rule 2) now lets the agent try again, as it was, an idea thrown
  away under an older as-of date, and the retry counts as a try. In simulation a retry was
  about as likely to be a lucky keep as a new idea (finding 8). Unproven with a real agent:
  whether it picks turn 3's idea up again after an advance (under 2026-07 it wins 95.17% of
  practice reshuffles). Never bundle near-misses and re-test them on the same validation
  months.
- Port the judge to the real load-forecasting data. For racing, score by log loss against the
  market's odds rather than by ROI.
- Nothing stops an advance during a night; add a guard only if it goes wrong.
- `scripts/night_sim.py`'s robot submits every idea that looks better on its practice period:
  it has no `SUBMIT_PCT`. So about 200 of its 300 ideas are judged, far more than the real
  loop would judge, and its try counts run higher than a real ledger's will.

## Practical notes
- uv manages Python 3.12 and the packages (`pyproject.toml`, `uv.lock`). Run everything with
  `uv run` (for example `uv run loop.py run 5`), not `python3`, and add packages with `uv add`.
  numpy, pandas and pyarrow are pinned to the sandboxes' versions: change them in
  `pyproject.toml` and both sandbox Dockerfiles together. scipy is only for the simulations.
- A fresh checkout has no data: run `uv run scripts/make_data.py` once. It makes the same
  days every time, and `loop.py` and `judge.py` stop with one line asking for it if the
  store is missing. Setting `TOY_DATA` to a folder points `store.py`, and so everything, at
  a different data folder; the tests use it, so each test file works on a temporary store
  of its own and never touches `data/`. `TOY_AS_OF` does the same for the as-of file: each
  test file holds its own as-of date in a temporary file, so the scores it expects don't
  depend on what `as_of.txt` says.
- `scripts/` holds helpers outside the main loop (`make_data.py`, `ingest.py`, `answer_key.py`, the simulations). Run them from the repo root.
- `tests/` holds the tests, which run with pytest. One command from the repo root runs them all:
  `uv run pytest` (about 3.5 minutes; Docker running, no token, no `make_data.py` first). One file:
  `uv run pytest tests/test_store.py`. Tests by name: `uv run pytest -k proxy`. `tests/conftest.py`
  holds what the test files share.
- GitHub runs the tests and the code checks on every pull request and every push to `main`
  (`.github/workflows/ci.yml`: the `pytest` job and the `checks` job). It needs no secret.
- Code checks read the code without running it. Before a commit, run the first three; the
  `checks` job runs all but the last, and that one runs by itself.

  | Check | Command | What it stops |
  |---|---|---|
  | Ruff lint | `uv run ruff check` (add `--fix` to repair what it can) | Likely mistakes; a function with no return type or no docstring |
  | Ruff format | `uv run ruff format` | Any layout but the formatter's, 100 characters wide |
  | Pyright | `uv run pyright` | A value used as the wrong type, such as one that may be None |
  | zizmor, actionlint | In `ci.yml` | Unsafe settings and mistakes in the workflow file |
  | hadolint | In `ci.yml` | Common Dockerfile mistakes |
  | Pin test | `uv run pytest tests/test_pins.py` | numpy, pandas or pyarrow differing between `pyproject.toml` and a sandbox Dockerfile |
  | gitleaks | By itself before each commit, after `uv run pre-commit install` once per checkout | A commit that holds a secret, such as the token in `.env` |

  The rules, and each rule left out with its reason, are in `pyproject.toml` (`[tool.ruff]`,
  `[tool.pyright]`), `.hadolint.yaml` and `.gitleaks.toml`. Ruff and pyright skip
  `lab/model.py` and `history/`: the agent writes those, and a kept idea must never fail a
  style check. Three docker argument lists sit between `# fmt: off` and `# fmt: on`, so each
  setting keeps its reason beside it. The actions in `ci.yml` are pinned to commit hashes:
  to update one, change the hash and the version in the comment beside it.
- Docker Desktop must be running. `uv run loop.py build` builds the three images (`run`,
  `advance` and `holdout` do it first anyway); `uv run pytest tests/test_sandboxes.py` checks the isolation
  (about 30 s).
- The agent sandbox can't use the Mac's Keychain login, so `loop.py run` needs a token: run
  `claude setup-token` once and `export CLAUDE_CODE_OAUTH_TOKEN=...` in the shell. The sandbox
  runs `claude -p --model sonnet` with Claude Code 2.1.319. Our agent turns cost about
  $0.11-0.16 each (Claude Code's own estimate).
- The proxy stays up after a run; `docker exec toy-agent-proxy cat /var/log/squid/access.log`
  lists every request the agent sent through it, until the next run or test run replaces
  the proxy. Attempts to go around the proxy fail without a trace.
- `uv run loop.py reset` returns to the baseline and to the as-of date the research started
  under, and clears the ledger and history.
- If you change a sandbox or the messages, run `tests/test_sandboxes.py`. Docker is a
  strong fence for an over-eager agent, not a guarantee against hostile code.
- If you change `turn()`, the ledgers, the try count or the reply schema, run
  `tests/test_loop.py` (about 1.5 minutes; real judge and model sandbox, a stand-in agent, no
  token needed). Its last section is the try count's table: 8 tries, then four real advances.
- `lab/practice.py` carries its own copy of the judge's walk and reshuffle, because the
  agent's workspace has no `judge/`. Its practice period is the newest 12 months mounted at
  `/data` (`PRACTICE_MONTHS`, the same setting as the judge's). Change them together;
  `tests/test_loop.py` checks that they print the same numbers, and `tests/test_as_of.py`
  checks it again after an advance.
- If you change the periods, `advance`, `reset`, `holdout` or the brief's template, run
  `tests/test_as_of.py` (about 1 minute; a temporary store, as-of file and lab). It also
  searches `loop.py`, `judge/`, `lab/` and `docker/` for written-out dates and fails if it
  finds one: write periods as rules there, and dates in ISO 8601 (`2025-12`) everywhere else.
- If you change `store.py`, `scripts/make_data.py` or `scripts/ingest.py`, run
  `tests/test_store.py` (about 30 s). It holds the SHA-256 of the old `judge/full_data.csv`,
  so the days 2023-01-01 to 2025-12-31 can't change unnoticed. `tests/test_size.py` scores a
  20-year, 303-column store; run it with `-s` to see the time printed (about 12 s for the
  validation period, one year).
- A model gets its `history` without a copy being made (pandas copy-on-write, switched on
  in `runner.py`, `lab/practice.py` and `lab/check.py`). Writing to `history` is safe: the
  next call still sees the original table. Known cost that stays: the runner rebuilds its
  table once per revealed day.

## How Hank likes explanations
When using math or formulas, give a plain-English interpretation, a small numeric example
with real numbers, and an analogy or intuition. Minimize symbols and define every variable.

## Background research
Published work on agent research loops: Karpathy's autoresearch, the five big pitfalls with
their evidence, and ways to reuse a validation set safely, with sources. Imported so it's always in
context:
@docs/research.md

## Communication
- Speak in simple, plain English whenever possible. Lead with non-technical framing (motivation, what changed, what success looks like) and keep math, code paths, and implementation detail in lower sections or appendices.
- Decision rules belong in tables with concrete numerical thresholds, not prose.

## Scratch space
- `scratch/` at the repo root is gitignored and is where anything the work needs but the history does not goes: ad-hoc scripts, probe output, query dumps, run logs, etc.
- Use it instead of `/tmp`.
- One directory per piece of work, named `<YYYY-MM-DD>-<slug>` — the date it was created, never renamed afterwards — with the issue number first in the slug when there is one: `scratch/2026-10-04-99-variance-inflation/`, `scratch/2026-10-04-thriftserver-oom/`. So `ls scratch/` reads oldest to newest, which is the only question ever asked of the folder ("what can go?"), and an issue's work is still found with `ls scratch/ | grep 99`. The date is the directory's birthday, not the date of the thing it is about.
- A dated directory that is worth keeping gets a one-line `README.md` saying what the question was — a month later the slug gives the topic and nothing else.
- A loose file at the top level (`scratch/probe.py`) means throwaway: it may be deleted at any time, by anyone. Anything that should survive goes in a dated directory. Nothing sweeps the folder automatically; pruning is by hand, by date.

## Git conventions
- Unless told otherwise, make the code changes for a PR in a separate git worktree under `.claude/worktrees/` (gitignored), not in the main checkout, which stays on `main` for other work.
- Branches follow [Conventional Branch](https://conventionalbranch.org/)
- Commits follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)

## Code clarity
- For Python code, always use NumPy-style docstrings (`Parameters` / `Returns` / `Raises` sections with the underlined-header format), not Google-style `Args:`.
- Use builtin types (`tuple`, `dict`, `list`, `set`) instead of typing module equivalents (`Tuple`, `Dict`, `List`, `Set`).
- Use `xxx | None` instead of `Optional[xxx]` for optional types.

## Agent skills

### Issue tracker

GitHub Issues on hankehly/toy-autoresearch, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root, created only when needed. See `docs/agents/domain.md`.
