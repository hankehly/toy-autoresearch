"""Proves that the isolation holds. Run it from the repo root:

    uv run pytest tests/test_sandboxes.py

The model sandbox: honest models must score exactly what the old judge gave them, and
models that try to cheat must crash instead of scoring. It starts with the months before
the scored period and no others, and a model's writes to its history don't last.
The agent sandbox: its only way out is the proxy, which lets through Claude's API alone.
It has the months up to the end of the practice period, read-only, and none after.
Only a plain model.py comes back from it, and a run without a login token stops at once.
The as-of date is held at 2025-12 in a temporary file (see as_of_file in conftest.py), so
the months and scores expected here stay right after the real one has advanced.
About 30 seconds.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import judge
import loop
import store

ROOT = Path(__file__).resolve().parent.parent

# every test here needs the toy's data in the store (conftest.py)
pytestmark = pytest.mark.usefixtures("toy_store")


def write_model(folder, code) -> Path:
    """Save a test model as <folder>/model.py and return its path."""
    path = folder / "model.py"
    path.write_text(code)
    return path


# --- The model sandbox: honest models score exactly as before ----------------------
# The scores are what the old judge gave these models, before the sandboxes existed.

SEVEN_DAY_AVERAGE_THAT_PRINTS = """
def fit(history) -> None:
    print("fitting")  # a print must never be mistaken for the model's reply
    return None


def predict(state, history, today) -> float:
    return history["load"].tail(7).mean()
"""

MONTH_AVERAGE_LEARNED_IN_FIT = """
def fit(history) -> float:
    return history["load"].tail(28).mean()


def predict(state, history, today) -> float:
    return state
"""

ANSWER_KEY = (ROOT / "scripts" / "answer_key.py").read_text()

# The runner hands model.py its history without copying it, so a write must stay in the table
# it was made to. If one reached the next call, these guesses would be 0 and not the 7-day average.
SEVEN_DAY_AVERAGE_THAT_WRITES_TO_ITS_HISTORY = """
def fit(history) -> None:
    history["dow"] = history["date"].dt.dayofweek  # a new column
    history.loc[:, "load"] = 0.0                    # and new values over the old ones
    return None


def predict(state, history, today) -> float:
    assert "dow" not in history.columns, "a column added in an earlier call is still there"
    guess = history["load"].tail(7).mean()
    history["dow"] = history["date"].dt.dayofweek
    history.loc[:, "load"] = 0.0
    return guess
"""


@pytest.mark.parametrize(
    "code, old_score",
    [
        pytest.param(SEVEN_DAY_AVERAGE_THAT_PRINTS, 5.782, id="7-day average that prints"),
        pytest.param(MONTH_AVERAGE_LEARNED_IN_FIT, 7.528, id="28-day average learned in fit"),
        pytest.param(ANSWER_KEY, 2.830, id="answer key"),
        pytest.param(
            SEVEN_DAY_AVERAGE_THAT_WRITES_TO_ITS_HISTORY,
            5.782,
            id="7-day average that writes to its history",
        ),
    ],
)
def test_an_honest_model_scores_as_before(code, old_score, tmp_path):
    misses = judge.daily_misses(write_model(tmp_path, code), "val")
    assert round(float(misses.mean()), 3) == old_score


# --- The model sandbox: cheats crash instead of scoring -----------------------------
# A crash in the sandbox reaches the judge as RuntimeError. Any other result means the
# cheat ran somewhere it could do harm.

HOST_STORE = "<the store's folder on your Mac>"  # replaced by the folder when the test runs
CHEATS = {
    "peeks at the judge's variables": """
import inspect


def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return inspect.currentframe().f_back.f_locals["day"].load  # today's answer, if it's there
""",
    "reads the answers from the store on your Mac": f'''
import pandas as pd

ANSWERS = pd.read_parquet("{HOST_STORE}/month=2024-07/part.parquet").set_index("date")["load"]


def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return ANSWERS[today["date"]]
''',
    "reads a validation month from /data": """
import pandas as pd

ANSWERS = pd.read_parquet("/data/month=2024-07/part.parquet").set_index("date")["load"]


def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return ANSWERS[today["date"]]
""",
    "reaches the internet": """
import urllib.request


def fit(history) -> None:
    urllib.request.urlopen("https://example.com", timeout=5)
    return None


def predict(state, history, today) -> float:
    return 100.0
""",
    "leaves a note for the next agent": """
def fit(history) -> None:
    with open(__file__, "a") as f:  # model.py is the one file the sandbox shares with your Mac
        f.write(f"# hidden days seen so far: {len(history)}\\n")
    return None


def predict(state, history, today) -> float:
    return 100.0
""",
}


@pytest.mark.parametrize("code", CHEATS.values(), ids=CHEATS.keys())
def test_a_cheat_is_stopped(code, tmp_path):
    cheat = write_model(tmp_path, code.replace(HOST_STORE, str(store.STORE)))
    with pytest.raises(RuntimeError):
        judge.daily_misses(cheat, "val")


# --- The model sandbox: it starts with the months before the scored period, and no others ---
# The months a sandbox can see are the month folders mounted into it. This model says which
# is its newest, as its guess: month=2024-06 comes back as 202406.

NAMES_ITS_NEWEST_MONTH = """
import os


def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return float(max(os.listdir("/data")).removeprefix("month=").replace("-", ""))
"""


@pytest.mark.parametrize(
    "period, newest", [("practice", 202306), ("val", 202406), ("holdout", 202506)]
)
def test_the_model_sandboxs_newest_month_is_the_one_before_the_first_scored_month(
    period, newest, tmp_path
):
    sandbox = judge.open_model_sandbox(write_model(tmp_path, NAMES_ITS_NEWEST_MONTH), period)
    guess = judge.request(sandbox, {"op": "predict", "today": {"date": "2000-01-01"}})["guess"]
    assert sandbox.stdin is not None  # open_model_sandbox() opens it as a pipe
    sandbox.stdin.close()
    sandbox.wait()
    assert guess == newest


# --- The agent sandbox: only a plain model.py comes back ----------------------------
# tmp_path stands in for the workspace the agent leaves behind.


def test_hand_off_no_model_py_nothing_comes_back(tmp_path):
    kept = tmp_path / "kept.py"
    assert loop.copy_submission(tmp_path, kept) is False
    assert not kept.exists()


def test_hand_off_a_shortcut_is_refused(tmp_path):
    kept = tmp_path / "kept.py"
    (tmp_path / "elsewhere.txt").write_text("some other file on your Mac")
    (tmp_path / "model.py").symlink_to(tmp_path / "elsewhere.txt")
    assert loop.copy_submission(tmp_path, kept) is False
    assert not kept.exists()


def test_hand_off_a_plain_model_py_comes_back(tmp_path):
    kept = tmp_path / "kept.py"
    (tmp_path / "model.py").write_text("# a plain model.py\n")
    assert loop.copy_submission(tmp_path, kept) is True
    assert kept.read_text() == "# a plain model.py\n"


# --- The agent sandbox: no login token, no run --------------------------------------
# Checked in a separate Python, so the test never starts a real agent turn.


def test_without_a_login_token_run_stops_and_says_how_to_get_one():
    no_token = {
        name: value for name, value in os.environ.items() if name != "CLAUDE_CODE_OAUTH_TOKEN"
    }
    stop = subprocess.run(
        [sys.executable, "-c", "import loop; loop.need_login_token()"],
        cwd=ROOT,
        env=no_token,
        capture_output=True,
        text=True,
    )
    assert stop.returncode != 0
    assert "claude setup-token" in stop.stderr


# --- The agent sandbox: its only way out is the proxy, open to Claude's API alone ----
# The probe runs inside the very sandbox the agent gets: loop.agent_sandbox().

PROBE = """
import json, os, subprocess, sys, urllib.error, urllib.request


def reach(url, through_proxy=True) -> str:
    no_proxy = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener() if through_proxy else urllib.request.build_opener(no_proxy)
    try:
        opener.open(url, timeout=10)
    except urllib.error.HTTPError:
        pass  # an error page is still an answer: the site was reached
    except Exception as error:
        return str(error)
    return "answered"


print(json.dumps({
    "claude": reach("https://api.anthropic.com/"),
    "github": reach("https://github.com/"),
    "github_around_the_proxy": reach("https://github.com/", through_proxy=False),
    "sees_the_repo": os.path.exists(sys.argv[1]),
    "months": sorted(os.listdir("/data")),
    # each month folder is its own mount. The mount's read-only flag is the proof: a file that
    # merely refuses writes could sit on a writable mount
    "months_are_read_only": all(os.statvfs(f"/data/{month}").f_flag & os.ST_RDONLY for month in os.listdir("/data")),
    "claude_version": subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout,
}))
"""


@pytest.fixture(scope="module")
def seen(toy_store, tmp_path_factory) -> dict:
    """Start the proxy, run the probe once in the agent sandbox, and return what it saw.

    Parameters
    ----------
    toy_store : None
        The toy's data, which the agent sandbox mounts.
    tmp_path_factory : pytest.TempPathFactory
        pytest's own fixture, which makes temporary folders.

    Returns
    -------
    dict
        The probe's answers, by the names in PROBE.
    """
    # A leftover network with a way out (made by hand, say) must not survive start_proxy().
    subprocess.run(["docker", "rm", "--force", loop.PROXY], capture_output=True)
    subprocess.run(["docker", "network", "rm", loop.NETWORK], capture_output=True)
    subprocess.run(
        ["docker", "network", "create", loop.NETWORK], check=True, stdout=subprocess.DEVNULL
    )

    loop.start_proxy()
    loop.start_proxy()  # a proxy left over from an earlier run is replaced, not tripped over

    work = tmp_path_factory.mktemp("agent-workspace")
    probe = subprocess.run(
        [*loop.agent_sandbox(work), "python", "-c", PROBE, str(ROOT)],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr[-300:]
    return json.loads(probe.stdout)


def test_the_agent_sandbox_reaches_claudes_api_through_the_proxy(seen):
    assert seen["claude"] == "answered"


def test_the_agent_sandbox_cant_reach_github_through_the_proxy(seen):
    assert "403" in seen["github"]


def test_the_agent_sandbox_cant_go_around_the_proxy(seen):
    assert seen["github_around_the_proxy"] != "answered"


def test_the_agent_sandbox_cant_see_the_repo_on_your_mac(seen):
    assert seen["sees_the_repo"] is False


def test_the_agent_sandbox_has_the_months_up_to_the_end_of_the_practice_period(seen):
    months = seen["months"]
    assert (len(months), months[0], months[-1]) == (18, "month=2023-01", "month=2024-06")


def test_the_agent_sandbox_cannot_read_any_month_after_the_practice_period(seen):
    assert len(store.months()) > 18  # the store does hold later months
    assert not any(month > "month=2024-06" for month in seen["months"])


def test_the_agent_sandbox_cant_change_the_months_it_has(seen):
    # every month folder is mounted read-only
    assert seen["months_are_read_only"] is True


def test_the_agent_sandbox_has_the_macs_claude_code_version(seen):
    assert "2.1.319" in seen["claude_version"]
