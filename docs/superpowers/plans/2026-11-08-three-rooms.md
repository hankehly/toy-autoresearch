# Three Rooms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep agent-written code away from the answers by running the agent, and later its
model.py, in two separate locked Docker containers ("rooms"). The host keeps the answers.

**Architecture:** Three rooms. The host runs only code a human wrote (loop.py, judge.py) and
holds all the data. The model room is a container with no network and no files except a
read-only model.py. The judge talks to it in one-line JSON "notes" and reveals each day's
actual load only after the model's guess. The agent room is a container that sees only the
turn's workspace. Its network's only exit is a "door" container (a squid proxy) that lets
through api.anthropic.com and nothing else.

**Tech Stack:** Python 3.12, numpy 2.2.3, pandas 2.2.3, Docker 29 (Docker Desktop), squid 6
on alpine 3.20, Claude Code 2.1.319.

**Spec:** the design Hank approved in chat on 2026-11-08 (his "three rooms" summary, with the
corrections we discussed). An earlier draft of the code is in session 36c84aa1's scratchpad
under `proposal/`. Checked before planning: the door lets api.anthropic.com through (HTTP
404) and refuses github.com (403), and a non-root container user can write to a host temp
folder.

## Global Constraints

- Clarity first: plain functions, short files, comments that say why. No config files,
  retries or logging frameworks.
- Both rooms use `python:3.12-slim` with `numpy==2.2.3 pandas==2.2.3`, the Mac's versions.
- The agent room has Claude Code `2.1.319`, the Mac's version, and runs as a non-root user.
- Names: images `toy-model-room`, `toy-agent-room` and `toy-agent-door`, built from
  `rooms/model-room`, `rooms/agent-room` and `rooms/agent-door`. The network is
  `toy-agent-net`, and the door's address is `http://toy-agent-door:3128`.
- Scores must not change. On dev, the old judge gives: 7-day average 5.782, 28-day average
  learned in fit 7.528, answer key 2.830.
- The keep rule doesn't change, so `scripts/night_sim.py` stays as it is.
- The login token is passed by name only (`-e CLAUDE_CODE_OAUTH_TOKEN`). Its value never
  appears in code, logs or chat.
- Tests are one plain script, `python3 scripts/test_rooms.py` (no pytest). It prints PASS or
  FAIL for each check and exits 1 if any check fails.
- No commits without Hank's OK.

## Review Focus

1. A model that prints must not have its print read as a guess. Test: the 7-day average with
   `print()` calls still scores 5.782 (Task 1).
2. Room images may be missing, or stale after a room file is edited. So `run`, `vault` and the
   test rebuild first; cached builds take seconds. Test: the test calls `loop.build()` first
   (Task 1).
3. The agent deletes model.py or leaves a shortcut (symlink) in its place. The turn is logged
   as "no model.py" and the night goes on. Test: `copy_submission` returns False for both
   (Task 2).
4. A door may be left over from an interrupted run. `open_door()` replaces it. Test: the test
   opens the door twice (Task 2).
5. No login token: `loop.py run` stops before doing anything and says how to make one. Test:
   run it with the variable unset (Task 2).

Not tested on purpose: `timeout 600` stops a model that hangs after 10 minutes, so a test
would take 10 minutes.

---

### Task 1: The model room

**Files:**
- Create: `rooms/model-room/Dockerfile`, `rooms/model-room/runner.py`
- Modify: `judge/judge.py`. Replace `load_model` and the in-process calls (lines 7 and
  25-47); `compare` and the command line stay as they are.
- Modify: `loop.py`. Add `build()` and a `build` command; `run` and `vault` call `build()`
  first.
- Create: `scripts/test_rooms.py` (the model-room checks)

**Interfaces:**
- Notes are one JSON object per line. From the judge to the room:
  - `{"op": "reveal", "days": [{"date": "2024-07-01", "temp_forecast": 23.7, "load": 110.3}, ...]}`
    gets no reply.
  - `{"op": "fit"}` gets `{"ok": true}`.
  - `{"op": "predict", "date": "2024-07-01", "temp_forecast": 23.7}` gets `{"guess": 104.6}`.
- `judge.py` provides:
  - `open_model_room(model_path) -> subprocess.Popen`
  - `tell(room, note) -> None`
  - `ask(room, note) -> dict`, which raises `RuntimeError("model.py crashed or ran out of time")`
    when the room goes silent
  - `daily_misses(model_path, split) -> np.ndarray`, with the same signature as today
- `loop.py` provides `build() -> None`, which builds `rooms/<name>` as `toy-<name>`.

- [ ] **Step 1: Write the model-room checks.** The test writes each model as a small file in a
  temp folder, then scores it with `judge.daily_misses(path, "dev")`.
  - Honest models must match the old judge to 3 decimals:
    - the 7-day average, with a `print()` in fit and predict: 5.782
    - the 28-day average learned in fit: 7.528
    - `judge/answer_key.py`: 2.830
  - Each cheat must raise `RuntimeError`:
    - peeks at the judge's variables: `inspect.currentframe().f_back.f_locals["day"].load`
    - reads the answer file: the absolute path to `judge/full_data.csv`
    - reaches the internet: `urllib.request.urlopen("https://example.com", timeout=5)` in fit
    - leaves a note for the next agent: writes `note.txt` next to model.py in fit
- [ ] **Step 2: Run it against today's judge.** Run `python3 scripts/test_rooms.py`. Expected:
  the honest checks PASS and all four cheats FAIL, because they score instead of crashing.
- [ ] **Step 3: Build the room and point the judge at it.**
  - `runner.py` keeps a growing table of the past. On `fit` it calls
    `model.fit(past.copy())`; on `predict` it calls `model.predict(state, past.copy(), today)`.
    It sends model.py's prints to stderr, so only replies reach the judge.
  - The Dockerfile ends with `ENTRYPOINT ["timeout", "600", "python", "/room/runner.py"]`.
  - `open_model_room` runs:
    `docker run --rm -i --pull never --network none --read-only --tmpfs /tmp --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges --memory 2g --cpus 2 --pids-limit 64 -v <model.py>:/room/model.py:ro toy-model-room`
  - `daily_misses` sends each predict note, waits for the guess, scores it, then reveals that
    day.
  - Add `loop.build()` to the top of the test.
  - The room's 600 s limit stays under loop.py's 900 s judge timeout, so a hung model ends as
    a crash, not a crashed night.
- [ ] **Step 4: Run the test.** Expected: every model-room check passes. Also check
  `python3 judge/judge.py compare lab/model.py history/turn_3.py 3`. Expected: `win_pct` 67.9
  and `keep` false, as in the ledger.
- [ ] **Step 5: Commit**, with Hank's OK.

### Task 2: The agent room and its door

**Files:**
- Create: `rooms/agent-room/Dockerfile`, `rooms/agent-door/Dockerfile`,
  `rooms/agent-door/squid.conf`
- Modify: `loop.py` (`AGENT`, `turn`, the `run` command)
- Modify: `scripts/test_rooms.py` (the agent-room checks)

**Interfaces:**
- `loop.py` provides:
  - `open_door() -> None`. It removes any old door, creates the internal network
    `toy-agent-net` if it's missing, starts `toy-agent-door` on that network, connects the
    door to `bridge`, and waits until it answers. The door stays up after a run, so you can
    read its log with `docker exec toy-agent-door cat /var/log/squid/access.log`.
  - `agent_command(work: Path) -> list[str]`
  - `copy_submission(work: Path, candidate: Path) -> bool`
- The whole of `squid.conf`:
  ```
  http_port 3128
  acl claude dstdomain api.anthropic.com
  http_access allow CONNECT claude
  http_access deny all
  ```

- [ ] **Step 1: Write the agent-room checks.**
  - Open the door twice (Review Focus 4).
  - Run a probe in the agent room on `toy-agent-net`, with
    `HTTPS_PROXY=http://toy-agent-door:3128`. These must hold:
    - Claude's API answers through the door (any HTTP status).
    - GitHub through the door is refused (403).
    - GitHub, going around the door, can't be reached.
    - The repo's path doesn't exist inside the room.
    - `claude --version` reports 2.1.319.
  - `copy_submission` returns False for a missing model.py and for a symlink, and True for a
    plain file.
  - `env -u CLAUDE_CODE_OAUTH_TOKEN python3 loop.py run 1` exits non-zero with a message that
    mentions `claude setup-token`, and creates no ledger row.
- [ ] **Step 2: Run it.** Expected: the new checks FAIL, because the images and functions
  don't exist yet.
- [ ] **Step 3: Build the room and the door, and move the agent in.**
  - The agent room image: `python:3.12-slim`, the same numpy and pandas pins, and curl. It adds
    the user `agent` and installs Claude Code with
    `curl -fsSL https://claude.ai/install.sh | bash -s 2.1.319`, with that user's
    `~/.local/bin` on PATH and `WORKDIR /work`.
  - The door image: alpine 3.20, squid and `squid.conf`, run with `CMD ["squid", "-N"]`.
  - `agent_command` runs:
    `docker run --rm --pull never --network toy-agent-net -e HTTPS_PROXY=http://toy-agent-door:3128 -e CLAUDE_CODE_OAUTH_TOKEN -e CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 -v <work>:/work toy-agent-room timeout 1800 claude -p ...`
    with today's flags plus `--disallowedTools WebSearch,WebFetch`. The time limit moves inside
    the room, because killing the docker client would leave the container running.
  - `turn` uses `copy_submission`. When it returns False, the turn is logged as "no model.py",
    which doesn't count as a try.
  - `run` checks for the token, builds, then opens the door.
- [ ] **Step 4: Run the whole test.** Expected: all checks pass.
- [ ] **Step 5: Commit**, with Hank's OK.

### Task 3: Docs, then one real turn

**Files:**
- Modify: `README.md`. Update "Run it", and replace "The lock is a toy lock" with "The three
  rooms" and the diagram.
- Modify: `CLAUDE.md`. Update "How it works" and the practical notes: Docker Desktop,
  `claude setup-token`, `python3 loop.py build` and `python3 scripts/test_rooms.py`. Drop
  the toy-lock note.
- Modify: `lab/program.md`, rule 3. Add that model.py runs in a locked room with no files and
  no internet.
- Create: `docs/three-rooms.drawio.png`, the diagram from chat retitled "The three rooms" and
  exported with the drawing embedded.

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Hank runs one real turn.** He runs `claude setup-token` and exports
  `CLAUDE_CODE_OAUTH_TOKEN` in his shell, then runs `python3 loop.py run 1`. Expected: one new
  ledger row and `history/turn_6.*`. The door's log shows `TCP_TUNNEL` lines for
  api.anthropic.com only.
- [ ] **Step 3: Commit**, with Hank's OK.
