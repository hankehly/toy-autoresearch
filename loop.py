"""The research loop.

Each turn:
  1. copy lab/ into a fresh, empty workspace, with the dates filled into program.md
  2. a fresh Claude Code session reads program.md + ledger.tsv and changes model.py. It works
     in the agent sandbox: a container that sees only that workspace and the months of the
     data store up to the end of the practice period, and reaches only Claude's API.
     It finishes with a structured reply: its idea, a lesson, its practice numbers, submit or not
  3. only model.py comes back, and the loop measures it against the champion on the practice
     period itself (the judge does the measuring, in the model sandbox)
  4. if the agent submitted it and it won enough of the practice reshuffles (SUBMIT_PCT),
     the judge compares it with the champion on the validation period, with a bar that
     rises with every idea tested (see judge.py)
  5. keep it (new champion) or throw it away, and write the verdict into both ledgers

    uv run loop.py run 5     # five more turns (continues from the ledger)
    uv run loop.py advance   # by hand, between nights: move the as-of date to the store's newest month
    uv run loop.py holdout   # score the holdout period once, at the end
    uv run loop.py reset     # back to the baseline, and to the as-of date the research started under
    uv run loop.py build     # build the Docker images (run, advance and holdout do this first anyway)

The data comes from the data store (store.py). `uv run scripts/make_data.py` builds it.
The periods (training, practice, validation, holdout) are counted back from the as-of date,
the one month in as_of.txt. The judge works them out: see periods() in judge.py.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jsonschema

import store

ROOT = Path(__file__).parent.resolve()
LAB, JUDGE, HISTORY = ROOT / "lab", ROOT / "judge", ROOT / "history"

# The as-of date: one line holding one month, the last month the research may use. The judge
# reads the same file. Only advance() and reset() write it. TOY_AS_OF names a different
# file; the tests use it to work on a temporary one.
AS_OF = Path(os.environ.get("TOY_AS_OF") or ROOT / "as_of.txt")

# Two ledgers. The full one holds every score and is never copied into the agent's workspace.
# The agent's copy is rewritten from it after every turn, without the validation columns: an
# agent that could read validation scores could pick and combine near-misses by them, which
# is tuning on the judge's period. Practice scores are safe to show: the agent has that period.
# as_of is the as-of date a row was measured under. Rows with different as-of dates were
# measured on different months, so their scores can't be compared directly.
LEDGER = ROOT / "ledger.tsv"
AGENT_LEDGER = LAB / "ledger.tsv"
COLUMNS = [
    "turn",
    "as_of",
    "idea",
    "practice_champion",
    "practice",
    "practice_wins",
    "val_mae",
    "win_pct",
    "bar_pct",
    "verdict",
    "lesson",
    "reply_check",
]
AGENT_COLUMNS = [
    "turn",
    "as_of",
    "idea",
    "practice_champion",
    "practice",
    "practice_wins",
    "verdict",
    "lesson",
]
TRIED = ("KEPT", "thrown away")  # the verdicts that count as a try, so they raise the bar
CHAMPIONS = ("start", "KEPT")  # the verdicts whose model became the champion
ADVANCED = "advanced"  # the verdict of the row an advance adds: the champion, measured again. Not a turn, not a try
AGENT_FILES = ["program.md", "model.py", "check.py", "practice.py", "ledger.tsv"]
# folders in docker/, each built as the Docker image toy-<name>
IMAGES = ["model-sandbox", "agent-sandbox", "agent-proxy"]
NETWORK = "toy-agent-net"  # the agent sandbox's network: its only way out is the proxy
PROXY = "toy-agent-proxy"  # the proxy's container: squid, which lets through Claude's API alone

# The agent's reply: a JSON object in this shape, not free text. Claude Code gets the schema
# through --json-schema and returns the object as "structured_output"; turn() checks the
# object against the same schema again, because a run can end without a valid one.
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "idea": {
            "type": "string",
            "minLength": 3,
            "maxLength": 100,
            "description": "The idea you tried, in under 12 words",
        },
        "lesson": {
            "type": "string",
            "minLength": 3,
            "maxLength": 300,
            "description": "One sentence: what this turn taught you",
        },
        "practice": {
            "description": "What practice.py printed for champion.py vs model.py, or null if you couldn't run it",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "champion_mae": {"type": "number", "exclusiveMinimum": 0},
                        "model_mae": {"type": "number", "exclusiveMinimum": 0},
                        "wins_pct": {"type": "number", "minimum": 0, "maximum": 100},
                    },
                    "required": ["champion_mae", "model_mae", "wins_pct"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ],
        },
        "submit": {
            "type": "boolean",
            "description": "true asks for model.py to be judged; false logs it as dropped on practice",
        },
    },
    "required": ["idea", "lesson", "practice", "submit"],
    "additionalProperties": False,
}

# An idea goes to the judge only if it wins at least this share (in %) of the practice
# reshuffles, by the loop's own measurement. The agent is told the same: the loop writes the
# number on top of the agent's ledger, and step 5 of program.md points there.
# Winning 90% of reshuffles means the gain is about 1.3 times its own luck (one standard
# deviation of a measured gain), whatever the units: if luck is 0.05 MWh, a gain of about
# 0.064 MWh. So the same setting fits a noisier or a quieter problem without retuning.
SUBMIT_PCT = 90

# The steps of a turn are in program.md, the brief you edit.
PROMPT = "Read program.md and follow its steps. Finish with the structured reply."
# the reply's shape; it comes back as "structured_output"
AGENT = [
    "timeout",
    "1800",  # a turn still running after 30 minutes is stopped
    "claude",
    "-p",
    PROMPT,
    "--model",
    "sonnet",
    "--permission-mode",
    "acceptEdits",  # may edit files in its own workspace only
    "--allowedTools",
    "Bash(python *),Bash(python3 *)",  # and run python; anything else is refused
    "--disallowedTools",
    "WebSearch,WebFetch",  # no web tools: its only sources are the lab files
    "--output-format",
    "json",
    "--json-schema",
    json.dumps(REPLY_SCHEMA),
]


def build() -> None:
    """Build the three Docker images. Quick when nothing has changed: Docker reuses its cache."""
    for name in IMAGES:
        subprocess.run(
            ["docker", "build", "--quiet", "-t", f"toy-{name}", str(ROOT / "docker" / name)],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def need_login_token() -> None:
    """Stop with instructions if there's no login token to pass into the agent sandbox.

    (Your Mac keeps its own Claude login in the Keychain, which a container can't use.)
    """
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        sys.exit(
            "The agent sandbox needs a Claude login token. In your shell, run:\n"
            "    claude setup-token\n"
            "    export CLAUDE_CODE_OAUTH_TOKEN=<the token it prints>\n"
            "then run this command again."
        )


def start_proxy() -> None:
    """Start the proxy: the only way out of the agent sandbox's network.

    It leads to Claude's API alone. It keeps running after a run, and its log lists every
    request the agent sent through it (the next run or test starts a fresh proxy, with an
    empty log)::

        docker exec toy-agent-proxy cat /var/log/squid/access.log
    """
    # remove an old proxy, if any
    subprocess.run(["docker", "rm", "--force", PROXY], capture_output=True)
    # and an old network, however it was made
    subprocess.run(["docker", "network", "rm", NETWORK], capture_output=True)
    # --internal: no way out of it but the proxy
    subprocess.run(
        ["docker", "network", "create", "--internal", NETWORK],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["docker", "run", "--detach", "--name", PROXY, "--network", NETWORK, "toy-agent-proxy"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # the proxy's other side: the internet
    subprocess.run(["docker", "network", "connect", "bridge", PROXY], check=True)
    for _ in range(
        50
    ):  # wait up to 5 seconds for squid to start listening (it takes about half a second)
        logs = subprocess.run(["docker", "logs", PROXY], capture_output=True, text=True)
        if "Accepting HTTP Socket connections" in logs.stdout + logs.stderr:
            return
        time.sleep(0.1)
    raise RuntimeError(f"the proxy didn't start; see: docker logs {PROXY}")


def as_of() -> str:
    """Read the as-of date.

    Returns
    -------
    str
        The month in as_of.txt, written as year-month.
    """
    return AS_OF.read_text().strip()


def periods() -> dict:
    """Ask the judge for the periods, which it counts back from the as-of date.

    Returns
    -------
    dict
        "as_of", and the months of "training", "practice", "val" and "holdout", oldest
        first, each written as year-month.

    Raises
    ------
    SystemExit
        If the judge can't work them out: the store doesn't reach the as-of date, or has no
        month before the practice period.
    """
    found = judge("periods")
    if "error" in found:
        sys.exit(found["error"])
    return found


def agent_months() -> list[str]:
    """List the months of the data store the agent has.

    Returns
    -------
    list[str]
        The training and practice periods: every month in the store up to the end of the
        practice period, oldest first. A month that arrived after the as-of date is not
        among them, and no sandbox gets it.
    """
    found = periods()
    return found["training"] + found["practice"]


def brief() -> str:
    """Fill the dates into the brief.

    lab/program.md is a template: it names no date itself, so it stays right when the as-of
    date advances.

    Returns
    -------
    str
        The text of lab/program.md with each {name} replaced: the first and last month of
        the agent's data and of the practice period, and the lengths of the practice and
        validation periods in months.
    """
    found = periods()
    months = found["training"] + found["practice"]  # the agent's months
    return (
        (LAB / "program.md")
        .read_text()
        .format(
            data_first=months[0],
            data_last=months[-1],
            practice_first=found["practice"][0],
            practice_last=found["practice"][-1],
            practice_months=len(found["practice"]),
            validation_months=len(found["val"]),
        )
    )


def agent_sandbox(work) -> list[str]:
    """The docker command that starts the agent sandbox, up to the image name.

    The sandbox sees only the turn's workspace and the agent's months of the data store, and
    its only way out is the proxy.
    """
    return [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        NETWORK,  # no internet, except through the proxy
        "-e",
        f"HTTPS_PROXY=http://{PROXY}:3128",  # so Claude Code sends its requests to the proxy
        "-e",
        "CLAUDE_CODE_OAUTH_TOKEN",  # your login token, passed on by name from your shell
        "-e",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",  # no update checks or error reports
        "-v",
        f"{work}:/work",  # the turn's workspace: the only folder it can change
        *store.mounts(agent_months()),  # the agent's months, read-only at /data
        "toy-agent-sandbox",
    ]


def run_agent(work) -> dict:
    """Run one agent session in the agent sandbox, on the workspace.

    Returns
    -------
    dict
        What Claude Code printed: a JSON object with the agent's "structured_output", its
        "result" text and the cost.
    """
    run = subprocess.run(agent_sandbox(work) + AGENT, capture_output=True, text=True)
    try:
        return json.loads(run.stdout)
    except json.JSONDecodeError:
        return {"result": run.stdout[-1000:] + run.stderr[-1000:]}


def check_reply(reported, measured) -> str:
    """Say whether the agent decided on the same practice numbers the ledger records.

    reported is the "practice" part of its reply; measured is the loop's own measurement
    (None if there is none).
    The loop's numbers are the record either way, so this never changes a verdict.
    """
    if reported is None:
        return "not run"  # the agent says it couldn't run practice.py
    if measured is None:
        return "-"  # nothing to compare with
    gaps = (
        abs(reported["champion_mae"] - measured["champion_mae"]),  # in MWh
        abs(reported["model_mae"] - measured["candidate_mae"]),
    )
    wins_gap = abs(reported["wins_pct"] - measured["win_pct"])  # in points
    # rounded, so floating-point error can't push a gap of exactly 0.001 over the line
    if round(max(gaps), 6) > 0.001 or round(wins_gap, 6) > 0.1:
        return "practice mismatch"
    return "ok"


def copy_submission(work, candidate) -> bool:
    """Bring back the agent's model.py, but only if it's a plain file.

    A shortcut (symlink) would be followed here on your Mac, so it could point at any file.
    """
    submitted = work / "model.py"
    if submitted.is_symlink() or not submitted.is_file():
        return False
    shutil.copy(submitted, candidate)
    return True


def judge(*args) -> dict:
    """Run the judge as a process of its own and read its answer.

    Parameters
    ----------
    *args
        The judge's command and what follows it, as on the command line: for example
        `judge("practice", champion, candidate)`.

    Returns
    -------
    dict
        What the judge printed, a JSON object. If the judge failed, `{"error": ...}` with
        the last line of its error text.
    """
    run = subprocess.run(
        [sys.executable, str(JUDGE / "judge.py"), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if run.returncode != 0:
        lines = run.stderr.strip().splitlines()
        return {"error": lines[-1] if lines else "judge failed"}
    return json.loads(run.stdout)


def read_ledger() -> list[dict[str, str]]:
    """The rows of the full ledger."""
    rows = LEDGER.read_text().strip().splitlines()[1:]
    # strict=False: a row whose last cells are empty has fewer cells than there are columns
    return [dict(zip(COLUMNS, r.split("\t"), strict=False)) for r in rows]


def ideas_judged() -> int:
    """How many ideas the judge has tested so far. The next one is tested against a higher bar."""
    return sum(r["verdict"] in TRIED for r in read_ledger())


def last_turn() -> int:
    """Find the newest turn in the ledger.

    Returns
    -------
    int
        Its number. An advance is not a turn, so its row has no number and is skipped.
    """
    return [int(r["turn"]) for r in read_ledger() if r["verdict"] != ADVANCED][-1]


def log(turn, idea, verdict, **cells) -> None:
    """Add a row to the full ledger and print it, then rewrite the agent's copy.

    cells are the row's other columns by name; those left out show as "-". Every row records
    the as-of date it was measured under.
    """
    cells = {"turn": turn, "as_of": as_of(), "idea": idea, "verdict": verdict, **cells}
    # one line of text per cell, no tabs
    row = {c: " ".join(str(cells.get(c, "-")).split()) or "-" for c in COLUMNS}
    with LEDGER.open("a") as f:
        f.write("\t".join(row.values()) + "\n")
    print(
        f"turn {turn}: {row['idea']} | practice {row['practice_champion']} -> {row['practice']}, won {row['practice_wins']}%"
        f" | validation MAE {row['val_mae']}, won {row['win_pct']}% (needed {row['bar_pct']}%) | {verdict}",
        flush=True,
    )
    write_agent_ledger()


def write_agent_ledger() -> None:
    """Rewrite the agent's copy from the full ledger.

    It holds practice numbers, verdicts and lessons, and no number measured on the validation
    period. The three comment lines on top say what the agent is up against: the champion,
    the bar the next idea must clear, and the share of practice reshuffles an idea needs
    before it is judged (SUBMIT_PCT).
    """
    rows = read_ledger()
    champion = [r for r in rows if r["verdict"] in CHAMPIONS][-1]
    # the champion's newest practice score: after an advance, the one in the advance's row
    measured = [r for r in rows if r["verdict"] in (*CHAMPIONS, ADVANCED)][-1]
    bar_pct = judge("bar", ideas_judged() + 1)["bar_pct"]
    lines = [
        f"# Champion: turn {champion['turn']}, practice {measured['practice']} MWh",
        f"# Ideas judged: {ideas_judged()}. The next must win {bar_pct}% of the judge's reshuffles.",
        f"# To be judged, an idea must win at least {SUBMIT_PCT}% of the practice reshuffles.",
        "\t".join(AGENT_COLUMNS),
    ]
    lines += ["\t".join(r[c] for c in AGENT_COLUMNS) for r in rows]
    AGENT_LEDGER.write_text("\n".join(lines) + "\n")


def champion_scores() -> dict[str, float]:
    """Measure the champion, lab/model.py, on the practice and validation periods.

    Returns
    -------
    dict[str, float]
        Its average miss in MWh on each: {"practice": ..., "val": ...}.

    Raises
    ------
    SystemExit
        If the judge can't score it on one of them.
    """
    scores = {period: judge("score", period, LAB / "model.py") for period in ("practice", "val")}
    errors = [score["error"] for score in scores.values() if "error" in score]
    if errors:
        sys.exit(f"The champion could not be measured: {errors[0]}")
    return {period: score["mae"] for period, score in scores.items()}


def start() -> None:
    """Start the research from the baseline.

    Save it as turn 0 and write its row, the first, to both ledgers. The baseline is measured
    before the ledger is made, so if it can't be measured, no half-made ledger is left behind.
    """
    scores = champion_scores()  # the baseline is the champion until an idea is kept
    HISTORY.mkdir(exist_ok=True)
    shutil.copy(LAB / "model.py", HISTORY / "turn_0.py")
    LEDGER.write_text("\t".join(COLUMNS) + "\n")
    log(
        0,
        "baseline: average of the last 7 days",
        "start",
        practice=scores["practice"],
        val_mae=scores["val"],
    )


def turn(t) -> None:
    """One turn of the loop: steps 1-5 at the top of this file."""
    # 1. copy the lab files into a fresh, empty workspace (a new temp folder), the brief with its dates filled in
    champion = LAB / "model.py"
    work = Path(tempfile.mkdtemp(prefix=f"agent-turn{t}-"))
    for name in AGENT_FILES:
        shutil.copy(LAB / name, work / name)
    (work / "program.md").write_text(brief())

    # 2. the agent changes model.py in the agent sandbox, and finishes with a structured reply
    began = time.time()
    result = run_agent(work)
    reply = result.get("structured_output")  # missing if the run ended without a valid one
    valid = jsonschema.Draft202012Validator(REPLY_SCHEMA).is_valid(reply)

    # 3. bring back model.py, delete the workspace, and save the turn in history/
    candidate = HISTORY / f"turn_{t}.py"
    if not copy_submission(work, candidate):  # only model.py comes back
        shutil.copy(champion, candidate)  # no plain model.py: the same as changing nothing
    shutil.rmtree(work)
    # the whole response, not picked fields: a run that fails has "errors" where "result" would be
    (HISTORY / f"turn_{t}.json").write_text(
        json.dumps({"seconds": round(time.time() - began), "response": result}, indent=2)
    )

    # the loop measures practice itself, so the ledger never depends on the agent's own numbers
    changed = candidate.read_bytes() != champion.read_bytes()
    # the loop's practice numbers; None if model.py didn't change or didn't run
    measured, crashed = None, False
    if changed:
        measured = judge("practice", champion, candidate)
        if "error" in measured:
            measured, crashed = None, True
    row = {}  # the row's cells so far
    if measured:
        row = {
            "practice_champion": measured["champion_mae"],
            "practice": measured["candidate_mae"],
            "practice_wins": measured["win_pct"],
        }

    # the verdicts that never reach the judge. None of them counts as a try, so the bar stays put
    if not valid:
        log(t, "-", "invalid reply", **row)  # the practice numbers, if any, are still logged
        return
    idea = reply["idea"]
    row["lesson"] = reply["lesson"]
    row["reply_check"] = check_reply(reply["practice"], measured)
    if not changed:
        log(t, idea, "no change", **row)
        return
    if crashed:
        log(t, idea, "crashed", **row)  # only the word: a model's error text could carry anything
        return
    # the loop goes by its own practice numbers, so a wrong number in the reply can't get an idea judged
    if not reply["submit"] or measured["win_pct"] < SUBMIT_PCT:
        log(t, idea, "dropped on practice", **row)
        return

    # 4. the judge compares it with the champion on the validation period; the bar rises with k
    k = 1 + ideas_judged()  # this is the k-th idea tested
    judged = judge("compare", champion, candidate, k)

    # 5. keep it (new champion) or throw it away, and write the verdict into both ledgers
    if "error" in judged:
        log(t, idea, "crashed", **row)
        return
    if judged["keep"]:
        shutil.copy(candidate, champion)
    log(
        t,
        idea,
        "KEPT" if judged["keep"] else "thrown away",
        val_mae=judged["candidate_mae"],
        win_pct=judged["win_pct"],
        bar_pct=judged["bar_pct"],
        **row,
    )


def advance() -> None:
    """Move the as-of date forward to the newest month in the data store.

    The months that have arrived since become part of the research. Run it by hand, between
    nights.

    The four periods move with the as-of date, by whole months: one or many, depending on
    how long it has been since the last advance. The champion is then measured on the new
    practice and validation periods, and one `advanced` row in the ledgers records the two
    scores. That row is not a try. No model changes, no old row is measured again, and
    nothing is scored on the holdout period.

    Raises
    ------
    SystemExit
        If the store has no month later than the as-of date, or the champion can't be
        measured. An advance happens whole or not at all: however it is cut short (the
        judge's error, a timeout, Ctrl-C), the as-of date and both ledgers are put back first.
    """
    old, new = as_of(), store.months()[-1]
    if new <= old:
        sys.exit(
            f"Nothing to advance to: the as-of date is {old}, and the data store's newest month is {new}."
        )
    if not LEDGER.exists():
        start()  # as `run` does: the baseline's row records the as-of date the research started under
    # the three files an advance writes
    before = {path: path.read_bytes() for path in (AS_OF, LEDGER, AGENT_LEDGER)}
    try:
        AS_OF.write_text(new + "\n")  # first, because the judge reads the as-of date from this file
        found = periods()
        scores = champion_scores()
        print(
            f"as-of date {old} -> {new}: "
            + ", ".join(
                f"{period} {found[period][0]} to {found[period][-1]}"
                for period in ("training", "practice", "val", "holdout")
            )
        )
        champion = [r for r in read_ledger() if r["verdict"] in CHAMPIONS][-1]
        log(
            "-",
            f"as-of date advanced from {old} to {new}: the champion (turn {champion['turn']}) measured again",
            ADVANCED,
            practice=scores["practice"],
            val_mae=scores["val"],
        )
    except (
        BaseException
    ):  # the champion can't be measured, a timeout or Ctrl-C: put all three files back as they were
        for path, content in before.items():
            path.write_bytes(content)
        print(f"Not advanced: the as-of date is still {old}.", file=sys.stderr)
        raise


def holdout() -> None:
    """Score the baseline and every kept model on the holdout period.

    The holdout period is the newest months, which no decision about any idea has used.
    Beside each holdout score goes the model's validation score, measured now on the current
    validation period: the val_mae in an old ledger row may come from other months.
    """
    found = periods()
    print(
        f"as-of date {found['as_of']}: validation is {found['val'][0]} to {found['val'][-1]}, "
        f"holdout is {found['holdout'][0]} to {found['holdout'][-1]}"
    )
    print(f"{'model':<58}{'val':>6}{'holdout':>9}")
    for r in read_ledger():
        if r["verdict"] in CHAMPIONS:
            model = HISTORY / f"turn_{r['turn']}.py"
            val_mae, holdout_mae = (
                judge("score", period, model)["mae"] for period in ("val", "holdout")
            )
            print(
                f"{'turn ' + r['turn'] + ': ' + r['idea']:<58.58}{val_mae:>6.2f}{holdout_mae:>9.2f}"
            )


def reset() -> None:
    """Go back to the baseline, and to the as-of date the research started under.

    That as-of date is the one in the ledger's `start` row. The ledgers and history/ are
    cleared. The data store is left alone, so months that have arrived stay.
    """
    AS_OF.write_text(read_ledger()[0]["as_of"] + "\n")
    shutil.copy(HISTORY / "turn_0.py", LAB / "model.py")
    shutil.rmtree(HISTORY)
    LEDGER.unlink()
    AGENT_LEDGER.unlink()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "build":
        build()
    elif cmd == "run":
        store.need()
        need_login_token()
        build()
        start_proxy()
        if not LEDGER.exists():
            start()
        last = last_turn()
        for t in range(last + 1, last + 1 + int(sys.argv[2] if len(sys.argv) > 2 else 5)):
            turn(t)
    elif cmd == "advance":
        store.need()
        build()
        advance()
    elif cmd == "holdout":
        store.need()
        build()
        holdout()
    elif cmd == "reset":
        reset()
