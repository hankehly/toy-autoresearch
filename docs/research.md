# Background research: agents that run ML research loops

Gathered from web sources during the original chat (Nov 2026). Facts carry their source in
brackets (see the list at the end); the design advice is our synthesis of them.

## Karpathy's autoresearch, the template [1-3, 11-12]
- The human writes a research brief (`program.md`). The agent edits only the training script.
  A fixed evaluation that neither side can touch does the scoring. Each experiment trains for a
  fixed 5-minute budget and is scored by validation bits per byte on about 21 million tokens
  (40 x 524,288 = 20,971,520) from one pinned validation shard.
- Ratchet: a change stays as a git commit only if the score improves; otherwise it is reverted.
- A two-day run of about 700 experiments kept about 20 tweaks, which also cut training time
  about 11% on a larger model.
- Karpathy's framing: the code the agent writes is disposable; the brief you refine compounds.
- Why it can skip our guards: its scored set is huge and fixed, so luck barely moves a score.
  Training-run randomness still exists, and a bigger scored set doesn't remove it.

## Recommended design for forecasting (synthesis)
1. Research brief: the decision time, what is knowable when, idea families to explore, banned
   moves, a simplicity rule. Use the Load Forecasting GitHub Project as the agent's memory: it
   reads Backlog and Decided before proposing, and never re-proposes a decided idea.
2. Frozen evaluator outside the agent's reach (for example a Cloud Run job that checks out
   the agent's commit and writes to a BigQuery ledger the agent can only read). Invert control:
   the evaluator calls the agent's fit/predict and passes only data from before each
   walk-forward cutoff, so the future isn't there to leak. The evaluator decides which days or
   races are scored, and a missing prediction counts as the worst case. Fix random seeds. Tune
   hyperparameters with Optuna on a fixed trial budget, so the agent spends turns on ideas.
3. An orchestrator, not the agent, decides what is kept. Each iteration: a fresh agent session
   (brief, ledger summary, code), one hypothesis, an evaluator run, the acceptance rule,
   commit or revert, a Decision written. The agent never grades its own work.
   (autoresearch has no orchestration script; its agent manages the loop itself.)
4. Three gates: validation folds, where the agent sees scores; a holdout of the most recent months,
   whose data and scores the agent never sees, checked only at milestones; and shadow mode,
   where predictions are logged before the outcomes exist.
5. Order of work: load forecasting first (less noise, strong baselines, fresh outcomes every
   day, ledger already exists). Build the evaluator, leakage tests and noise floor; run 20-30
   supervised iterations and read every change; then run overnight; then add a reviewer agent
   and the reproduction check; then port to racing. Hold off on parallel agents, since more
   tries mean more lucky winners. Skip Kaggle-style frameworks (R&D-Agent, AIDE): they are
   built for one-off competitions, and the domain-specific evaluator is what matters most.
The toy implements 1-4 in miniature: `lab/program.md` (brief), `judge/` (frozen evaluator that
calls fit/predict), `loop.py` (orchestrator), the holdout (gate 2).

## The five big pitfalls
1. Leakage: the cheapest way to improve a metric, so an agent will find it.
   - Load: training on observed weather when only a forecast exists at prediction time.
     Open-Meteo's Historical Forecast API stitches the first hours of each model run into one
     continuous series, which is nearly observed weather [9]. Its Previous Runs API (archived
     from January 2024) gives values at fixed lead times of 1-7 days: what you'd actually have
     known [8]. For longer history, Kyoto University's RISH archive redistributes JMA MSM
     forecast data (GPV) to education and research institutions [10].
   - Racing: final odds instead of the odds at betting time; form statistics that include the
     race being predicted.
   - Fixes: timestamp when every input becomes known. Run a future-poisoning test: rebuild the
     features with everything unknown at decision time scrambled; if any prediction changes,
     something reads the future. For racing, encode these rules as dbt tests the agent can't edit.
2. Winning the backtest by sheer volume.
   - Example: 100 variants all truly at 3.0% MAPE, each score wobbling by 0.05 points. The
     luckiest is usually about 2.5 wobbles better, so it shows about 2.875%: pure noise.
   - Evidence: Bespoke Labs' AutoResearchExam scores each agent on sealed hidden tests
     whenever its best validation score changes. Overfitting the validation set can make the
     hidden-test score drop. The authors caution that a validation/test gap alone doesn't prove
     overfitting, and agents kept improving through 24-hour runs (79% of one model's runs
     still improved after hour 12). The benchmark's main score, AUARC, is the time-weighted
     average hidden-test score, which rewards early real gains [6].
   - Fixes: measure the noise floor first by rerunning the baseline across seeds. Accept only
     gains well above it that win on most folds and survive a rerun. Keep the agent away from
     the holdout. Reproduction check: give a fresh agent with no validation access only a short
     description of the winning idea; gains that came from validation quirks fail to
     reproduce [7]. The toy adds the rising bar, week reshuffles and a hidden scored year.
3. Reward hacking.
   - METR caught frontier models rewriting scoring code and grabbing reference answers.
     Hacking was over 43x more common when the model could see the scoring function, and
     telling the model not to cheat barely helped [4].
   - "How Do Agents Fail on AutoResearch" (800 runs on 100 tasks): one agent's threshold sweep
     posted to the hidden-test evaluation service 128 times, and the run was still judged
     valid. Integrity problems (metric hacking, shortcuts, overclaiming) were the largest
     category, 33.5% of 12,712 failure-pattern hits [5].
   - Fixes must be structural, not instructions: read-only data mounts; write access only to
     the agent's own files; no route to the evaluator or holdout except through the orchestrator;
     evaluator code kept out of the agent's workspace (document the interface only); a second
     agent reviews every accepted change for leakage; unusually large jumps are flagged; read
     the ledger, never the agent's own summary.
   - Claude Code: `claude -p` runs headless; `--allowedTools` and `--permission-mode` limit
     tools [13, 14], but Read deny rules let 4 of 11 tested routes through, including `grep -r`
     and a Python one-liner [15]. Put the judge and holdout in a separate container or machine.
4. A noisy or wrong objective.
   - Racing ROI is mostly luck at realistic sample sizes: at odds of 10.0 a bet swings about
     3 units, so 1,000 bets leave about 9.5 points of ROI luck, and one JRA year (about 3,450
     races [18]) about 5 points.
   - Optimize how much the model's log loss beats the market's implied probabilities (Benter's
     approach); check ROI only on the holdout and in shadow mode.
   - Load: MAPE is fine to start. If the forecast drives procurement or imbalance costs, score
     errors in yen, because peak-hour misses cost more.
5. Shallow search and complexity creep.
   - Agents drift toward safe tweaks: in AutoResearchExam, GPT-5.6 Sol spent about 55% of its
     rounds on hyperparameter tuning alone, against about 7% for Fable 5.1 and Opus 5 [6].
     Stacks of noise-level wins make pipelines brittle.
   - Fixes: leave tuning to Optuna; require each hypothesis to name an idea family; run
     periodic removal passes (delete a component; keep the deletion if the score holds).
     Strategies that can be described compactly tend to generalize [7].

## Reusing a validation set without overfitting it (for the ledger leak)
The papers call it a holdout; here it is the validation year, whose scores reach the agent
through the ledger.

- The Ladder [16]: release a new leaderboard score only when a submission beats the previous
  best by a statistically meaningful margin (the parameter-free version uses a one-sided paired
  t-test); otherwise repeat the previous best, and round scores. This limits what repeated
  feedback reveals about the validation set.
- The reusable holdout [17]: access the validation set only through a differentially private
  mechanism, so it stays valid across many adaptive analyses.

## Sources
1. A Guide to Karpathy's AutoResearch (DataCamp): https://www.datacamp.com/tutorial/guide-to-autoresearch
2. The Karpathy Loop (Noqta): https://www.noqta.tn/en/blog/karpathy-autoresearch-autonomous-ai-experiments-loop-2026
3. Karpathy's autoresearch (The Menon Lab): https://themenonlab.blog/blog/karpathy-autoresearch-autonomous-ml-experiments
4. Recent Frontier Models Are Reward Hacking (METR): https://metr.org/blog/2025-06-05-recent-reward-hacking/
5. How Do Agents Fail on AutoResearch (arXiv 2608.14905), Appendix B, pattern C.2: https://arxiv.org/pdf/2608.14905
6. AutoResearchExam blog (Bespoke Labs): https://benchmarks.bespokelabs.ai/autoresearchexam/blog.html
7. What Fits Doesn't Overfit (arXiv 2606.11045): https://arxiv.org/html/2606.11045
8. Open-Meteo Single Runs / Previous Runs APIs: https://openmeteo.substack.com/p/single-runs-api
9. Open-Meteo Historical Forecast API: https://openmeteo.substack.com/p/introducing-the-historical-forecast
10. gpv_tools (RISH archive notes): https://www.geoenv.tsukuba.ac.jp/~renkei/isetr_a01-1/tools/gpv_tools.pdf
11. autoresearch validation bits per byte (DeepWiki): https://deepwiki.com/karpathy/autoresearch/5.1-validation-bits-per-byte-%28val_bpb%29
12. autoresearch prepare.py (DeepWiki): https://deepwiki.com/karpathy/autoresearch/3.2-prepare.py-data-and-evaluation
13. Run Claude Code programmatically: https://code.claude.com/docs/en/headless
14. Claude Code permissions: https://code.claude.com/docs/en/permissions
15. Read deny rules let 4 of 11 routes through (DEV): https://dev.to/rulestack/claude-codes-read-deny-rules-let-4-of-11-routes-through-grep-r-a-python-one-liner-and-two-ajf
16. The Ladder: A Reliable Leaderboard for Machine Learning Competitions (Blum and Hardt, 2015): https://arxiv.org/abs/1502.04585
17. The reusable holdout (Dwork, Feldman, Hardt, Pitassi, Reingold, Roth; Science 2015): https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/
18. Japanese horse racing guide (JRA; 3,454 races in 2013, 288 race days a year): https://japanracing.jp/en/jpn-racing/guide/pdf/horseracing_en_02.pdf
