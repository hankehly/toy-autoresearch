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
- Months. Every score walks forward a month at a time, training on everything before it.
  Jan-Jun 2023: warm-up. Jul 2023-Jun 2024: the agent's practice year.
  Jul 2024-Jun 2025: the judge's scored year (dev); the agent never has it.
  Jul-Dec 2025: the vault, opened once at the end.
- Keep rule: reshuffle the 52 scored weeks 10,000 times. The k-th idea tested must win at
  least 1 - 5%/k of reshuffles (95% for the 1st, 99.5% for the 10th, 99.95% for the 100th),
  which keeps the expected number of lucky keeps under one for a 300-idea night.
- `judge/answer_key.py` is the recipe that made the data (dev 2.83, vault 2.77 MWh). It is
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
   year and submitted the best; it looked better there and worse on the vault, and the keep
   rule threw it out.
2. Simulated 300-idea nights (half setting tweaks, half pure-noise inputs), old rules: about
   9 flukes kept per night; the ledger claimed +0.45 while the truth was +0.22, and real
   progress stopped after about 30 ideas. From an already-optimal model: 12 noise inputs
   kept, ledger +0.21, truth -0.03. Hiding the scored year cut flukes to under 1 per night;
   adding the rising bar made it 0 (ledger +0.24, truth +0.20), at a small cost in real gains.
3. Neighbouring days are only weakly linked in this toy (about +0.1), so reshuffling whole
   weeks barely matters here. It is kept for real data (weather spells, same-day race conditions).
4. Run 2, new rules (the current ledger and history): turn 1 went straight to a weekday plus
   heating/cooling regression (dev 2.92, vault 2.75), and the next 4 ideas were rejected.
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

## Open threads
- Allow practice-year bundles: let the agent submit several ideas that each helped on its
  practice year as one try, and see whether it captures turn 3's real gain.
- Remaining leak: the agent reads every rejected idea's dev score in the ledger. For long
  runs, show it only kept / thrown away.
- Set near-misses aside and re-test them on new months; never bundle them and re-test on the
  same scored year.
- Port the judge to the real load-forecasting data. For racing, score by log loss against the
  market's odds rather than by ROI.

## Practical notes
- Python 3 with numpy, pandas and scipy (`scripts/night_sim.py` and `scripts/bundle_sim.py` need scipy).
- `scripts/` holds helpers outside the main loop (`make_data.py`, the simulations). Run them from the repo root.
  Use `python3` on macOS.
- `loop.py` calls `claude -p --model sonnet`, so Claude Code must be installed and logged in.
  Our agent turns cost about $0.11-0.16 each (Claude Code's own estimate).
- `python3 loop.py reset` returns to the baseline and clears the ledger and history.
- The lock is a toy lock: the agent's Python could read `judge/` on this machine. For real
  work, run the judge and the vault in a separate container.

## How Hank likes explanations
When using math or formulas, give a plain-English interpretation, a small numeric example
with real numbers, and an analogy or intuition. Minimize symbols and define every variable.
