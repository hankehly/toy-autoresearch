# Toy autoresearch

An agent improves a forecasting model one idea per turn, and a judge it cannot reach decides
what is kept. This glossary fixes the words used for the data and how it moves over time.

## Language

### Models

**Baseline**:
The model the research started from, before any idea from the agent.
_Avoid_: starting model, turn 0

**Champion**:
The best model so far: the last idea the judge kept, or the baseline if none has been kept.
_Avoid_: current model, best model

### The loop

**Turn**:
One agent's work on one idea, from reading the brief to the verdict on what it changed.
_Avoid_: run, iteration, round, experiment

**Batch**:
The turns run one after another with no person stepping in between them. No rule depends
on where a batch ends: what a try counts depends on the as-of date alone.
_Avoid_: night, session, run, bundle

### Scoring

**Try**:
An idea the judge compared with the champion on the validation period.
_Avoid_: attempt, submission, test

**Try count**:
How many tries count against the current validation period. A try counts for less as the
months it was judged on leave that period.
_Avoid_: k, number of ideas

**Bundle**:
Several ideas submitted together as one try.
_Avoid_: batch, stack, combination

**Holdout score**:
A model's average miss on the holdout period.
_Avoid_: report card, test score

### Periods

**Period**:
A run of whole months that plays one role in the research. The four periods do not overlap
and are counted back from the as-of date.
_Avoid_: split, window, fold

**Training period**:
The months before the practice period. Every model learns from them and nothing is scored on them.
_Avoid_: warm-up

**Practice period**:
The months whose answers the agent has, where it scores its own ideas before submitting one.
_Avoid_: dev set

**Validation period**:
The months the judge scores to decide whether an idea is kept. The agent never has them.
_Avoid_: dev, scored year, test set

**Holdout period**:
The newest months, which no decision about any idea has used.
_Avoid_: vault, test set

### Time

**As-of date**:
The last month the research is allowed to use. It stays fixed until a person advances it.
_Avoid_: cutoff, snapshot date

**Advance**:
To move the as-of date forward so that months which have arrived since become part of the research.
_Avoid_: roll, slide, shift
