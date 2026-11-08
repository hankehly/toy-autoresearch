# Toy autoresearch

Context for Claude Code. This project started in a Claude chat (Nov 5-8, 2026); this file
carries over what was decided and learned there. README.md describes the files and commands.

## Goal
Hank wants AI agents to run the iterative research and model-building for predictive
problems that use traditional ML, such as next-day electricity load and horse racing. This
folder is the smallest working version: a made-up town's daily electricity use, an agent that
improves `lab/model.py` one idea per turn, and a frozen judge that decides what's kept.
The lessons are meant to carry over to his real load-forecasting project (a GitHub Project
used as a research ledger, with kept / rejected / set aside decisions) and his horse racing
platform (PySpark, dbt, LightGBM, 20+ years of races).

## How it works
- `loop.py run N`: each turn copies `lab/` to a temp folder, where a fresh `claude -p` agent
  reads `program.md` and `ledger.tsv`, changes only `model.py`, and checks it with `check.py`.
  Only `model.py` comes back. `judge/judge.py` compares it with the champion, and it is kept
  or thrown away and logged in `lab/ledger.tsv`.
- Three rooms (`rooms/`, Docker). The host runs only code a human wrote (`loop.py`,
  `judge.py`) and holds all the data. The agent works in the agent room, a container that
  sees only the turn's temp folder; its network's only way out is the door, a squid proxy
  that lets through api.anthropic.com alone. The judge runs `model.py` in the model room
  (no network, read-only, nothing but `model.py`) and talks to it in one-line JSON notes
  (reveal, fit, predict), revealing each day's load only after the guess.
  `scripts/test_rooms.py` proves the walls hold.
- Months. Every score walks forward a month at a time, training on everything before it.
  Jan-Jun 2023: training. Jul 2023-Jun 2024: the agent's practice year.
  Jul 2024-Jun 2025: the judge's scored year (validation); the agent never has it.
  Jul-Dec 2025: holdout, scored once at the end.
- Keep rule: reshuffle the 52 scored weeks 10,000 times. The k-th idea tested must win at
  least 1 - 5%/k of reshuffles (95% for the 1st, 99.5% for the 10th, 99.95% for the 100th),
  which keeps the expected number of lucky keeps under one for a 300-idea night.
- `judge/answer_key.py` is the recipe that made the data (validation 2.83, holdout 2.77 MWh). It is
  close to the best possible but not a hard floor: adding yesterday's temperatures beats it
  slightly, because that sharpens the estimate of today's actual temperature.
- `scripts/night_sim.py` re-implements the judge for speed (it reproduces the judge's scores exactly)
  and simulates whole nights with a random-tweak robot. True error is measured on 100
  parallel towns: the same recipe with fresh weather and noise. If you change the judge's
  rules, change `scripts/night_sim.py` to match.
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
7. Before the rooms, the judge ran `model.py` in its own process, next to the answers. A
   6-line model that read today's load from the judge's loop variable was kept as the 6th
   idea (won 100%; validation 2.14 and holdout 2.06 with noise added, 0.00 without), and it passed
   `check.py`. The holdout agreed, so every statistical guard failed at once. Walls, not
   statistics, stop this; the model room now makes it crash (`scripts/test_rooms.py`).

## Open threads
- Allow practice-year bundles: let the agent submit several ideas that each helped on its
  practice year as one try, and see whether it captures turn 3's real gain.
- Remaining leak: the agent reads every rejected idea's validation score in the ledger. For long
  runs, show it only kept / thrown away.
- Set near-misses aside and re-test them on new months; never bundle them and re-test on the
  same scored year.
- Port the judge to the real load-forecasting data. For racing, score by log loss against the
  market's odds rather than by ROI.

## Practical notes
- uv manages Python 3.12 and the packages (`pyproject.toml`, `uv.lock`). Run everything with
  `uv run` (for example `uv run loop.py run 5`), not `python3`, and add packages with `uv add`.
  numpy and pandas are pinned to the rooms' versions: change them in `pyproject.toml` and both
  room Dockerfiles together. scipy is only for the simulations.
- `scripts/` holds helpers outside the main loop (`make_data.py`, the simulations). Run them from the repo root.
- Docker Desktop must be running. `uv run loop.py build` builds the three images (`run` and
  `holdout` do it first anyway); `uv run scripts/test_rooms.py` checks the walls (about 30 s).
- The agent room can't use the Mac's Keychain login, so `loop.py run` needs a token: run
  `claude setup-token` once and `export CLAUDE_CODE_OAUTH_TOKEN=...` in the shell. The room
  runs `claude -p --model sonnet` with Claude Code 2.1.319. Our agent turns cost about
  $0.11-0.16 each (Claude Code's own estimate).
- The door stays up after a run; `docker exec toy-agent-door cat /var/log/squid/access.log`
  lists every request the agent sent through it, until the next run or test run replaces
  the door. Attempts to go around the door fail without a trace.
- `uv run loop.py reset` returns to the baseline and clears the ledger and history.
- If you change a room or the notes, run `scripts/test_rooms.py`. Docker is a strong fence
  for an over-eager agent, not a guarantee against hostile code.

## How Hank likes explanations
When using math or formulas, give a plain-English interpretation, a small numeric example
with real numbers, and an analogy or intuition. Minimize symbols and define every variable.

## Background research
Published work on agent research loops: Karpathy's autoresearch, the five big pitfalls with
their evidence, and ways to reuse a validation set safely, with sources. Imported so it's always in
context:
@docs/research.md
