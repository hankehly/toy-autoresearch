"""Proves that the data store holds the days it should, and that new months arrive only
through ingest. Run it from the repo root:

    uv run tests/test_store.py

Everything here happens in a temporary data folder (TOY_DATA points the store there), so
data/ is never touched. The scores come from the real judge and the real model sandbox.
Prints PASS or FAIL for each check, and exits with 1 if any check failed (about 30 seconds).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="test-store-"))
os.environ["TOY_DATA"] = str(TMP / "data")  # set before store is imported; the scripts and the judge started below follow it too
sys.path.insert(0, str(ROOT))
import loop  # noqa: E402  (imported after the path is set: these files live in the repo root, not here)
import store  # noqa: E402

loop.build()  # the judge runs every model in the model sandbox
failures = 0


def check(name, passed, detail="") -> None:
    """Print PASS or FAIL for one check, and count the failures.

    Parameters
    ----------
    name : str
        What the check proves.
    passed : bool
        Whether it held.
    detail : object, optional
        What was seen, printed after the name.
    """
    global failures
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def run(*command) -> subprocess.CompletedProcess:
    """Run one of the repo's scripts from the repo root, the way a person would, but with no
    login token: nothing here may start a real agent turn.

    Parameters
    ----------
    *command : str
        The script and its arguments, as typed after `uv run`.

    Returns
    -------
    subprocess.CompletedProcess
        The finished run, with what it printed.
    """
    no_token = {name: value for name, value in os.environ.items() if name != "CLAUDE_CODE_OAUTH_TOKEN"}
    return subprocess.run([sys.executable, *command], cwd=ROOT, env=no_token, capture_output=True, text=True)


def sha256_as_csv(days) -> str:
    """Hash days the way the old CSV file held them, to compare them with that file.

    Parameters
    ----------
    days : pd.DataFrame
        One row per day: date, temp_forecast, load.

    Returns
    -------
    str
        The SHA-256 of the days written as CSV text, as `shasum -a 256` prints it.
    """
    as_csv = days.assign(date=days.date.dt.strftime("%Y-%m-%d")).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(as_csv.encode()).hexdigest()


def val_scores() -> dict[str, float]:
    """Score the two models whose scores we know, with `judge.py score val`.

    Returns
    -------
    dict[str, float]
        Each model's average miss on the validation period, in MWh, by the model's name.
    """
    return {name: json.loads(run("judge/judge.py", "score", "val", path).stdout or "{}").get("mae")
            for name, path in [("baseline", baseline), ("answer key", ROOT / "scripts" / "answer_key.py")]}


baseline = TMP / "baseline.py"
baseline.write_text('''
def fit(history) -> None:
    return None


def predict(state, history, today) -> float:
    return history["load"].tail(7).mean()
''')


# --- No store yet: the judge and the loop say how to make one -------------------------------

for command in (["judge/judge.py", "score", "val", str(baseline)], ["loop.py", "run", "1"], ["loop.py", "holdout"]):
    stopped = run(*command)
    check(f"without a store, {' '.join(command[:2])} stops with one line that says to run make_data.py",
          stopped.returncode != 0 and stopped.stdout == "" and len(stopped.stderr.strip().splitlines()) == 1
          and "uv run scripts/make_data.py" in stopped.stderr, stopped.stderr.strip()[-200:])


# --- make_data.py: a pretend future, and a store that starts where the CSV files ended ---

made = run("scripts/make_data.py")
check("make_data.py runs", made.returncode == 0, made.stderr[-300:])

future, stored = store.months(store.FUTURE), store.months()
check("the pretend future is 2023-01 to 2042-12, one folder per month",
      (len(future), future[0], future[-1]) == (240, "2023-01", "2042-12"), (len(future), future[:1], future[-1:]))
check("the store starts with 2023-01 to 2025-12",
      (len(stored), stored[0], stored[-1]) == (36, "2023-01", "2025-12"), (len(stored), stored[:1], stored[-1:]))
check("a month folder is laid out the way Spark writes it: month=2023-01/part.parquet",
      (store.STORE / "month=2023-01" / "part.parquet").is_file())

# The ledger, history/ and the findings in CLAUDE.md describe the 1,096 days that used to be
# judge/full_data.csv. This is that file's SHA-256 (see it with: git show d7be1ad:judge/full_data.csv | shasum -a 256).
OLD_CSV_SHA256 = "c8222e82de68e38b777e9c4a1bd5e30c83db93ee37d6c1b1886ff4e27939f506"
original_days = store.read(stored)
check("the store's 1,096 days are the old judge/full_data.csv, value for value",
      len(original_days) == 1096 and sha256_as_csv(original_days) == OLD_CSV_SHA256,
      f"{len(original_days)} days, columns {list(original_days.columns)}")

january = store.read(["2026-01"], store.FUTURE)
check("the days after 2025 are whole months of the same three columns",
      len(january) == 31 and list(january.columns) == ["date", "temp_forecast", "load"]
      and january.date.dt.strftime("%Y-%m").eq("2026-01").all() and january.notna().all().all(),
      f"{len(january)} days, columns {list(january.columns)}")

# The scores the judge gave before the store existed: turn 0 in ledger.tsv, and the answer key.
KNOWN_SCORES = {"baseline": 5.782, "answer key": 2.83}
check("the judge scores the validation period from the store as it did from the CSV file",
      val_scores() == KNOWN_SCORES, val_scores())


# --- ingest.py: time passes a whole month at a time ---------------------------------------

def file_bytes(folder) -> dict[str, bytes]:
    """Read every file under a folder, to compare the folder before and after.

    Parameters
    ----------
    folder : Path
        The folder to read.

    Returns
    -------
    dict[str, bytes]
        Each file's contents, by its path inside the folder.
    """
    return {str(path.relative_to(folder)): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()}


store_before, future_before = file_bytes(store.STORE), file_bytes(store.FUTURE)
ingested = run("scripts/ingest.py", "--through", "2026-06")
stored = store.months()
check("ingest --through 2026-06 adds the six months the store didn't have",
      ingested.returncode == 0 and (len(stored), stored[-1]) == (42, "2026-06"), ingested.stderr[-300:] or stored[-7:])
check("ingest: an added month is the future's month, day for day",
      store.read(["2026-06"]).equals(store.read(["2026-06"], store.FUTURE)))
store_after = file_bytes(store.STORE)
check("ingest never changes a month the store already had",
      all(store_after[name] == content for name, content in store_before.items()))
check("ingest touches nothing else: the future is as it was, and the data folder holds only the two folders",
      file_bytes(store.FUTURE) == future_before and sorted(p.name for p in store.DATA.iterdir()) == ["future", "store"])

run("scripts/ingest.py", "--through", "2026-03")
check("ingest with an older month adds nothing and removes nothing", file_bytes(store.STORE) == store_after)

check("after ingest, the judge's scores are the same: nothing reads the new months yet",
      val_scores() == KNOWN_SCORES, val_scores())


# --- A write that fails must not leave a month that looks finished ---------------------------

class DaysThatFailHalfway:
    """Stands in for a month's days when the disk fills up: half a file is written, then the write fails."""

    def to_parquet(self, path, index) -> None:
        """Write half a file where a table would write its Parquet file, then fail.

        Parameters
        ----------
        path : Path
            Where store.write() asked for the file.
        index : bool
            Unused. A table's to_parquet() takes it.

        Raises
        ------
        OSError
            Always, as a full disk would.
        """
        Path(path).write_bytes(b"half a file")
        raise OSError("No space left on device")


def write_that_fails(month) -> None:
    """Try to write a month to the store with a write that fails halfway.

    Parameters
    ----------
    month : str
        The month, written like "2026-07".
    """
    try:
        store.write(month, DaysThatFailHalfway())
    except OSError:
        pass


write_that_fails("2026-07")  # a month the store doesn't have yet
check("a write that fails for a new month adds no month", store.months()[-1] == "2026-06", store.months()[-1])
run("scripts/ingest.py", "--through", "2026-07")
check("ingest completes such a month the next time",
      store.months()[-1] == "2026-07" and store.read(["2026-07"]).equals(store.read(["2026-07"], store.FUTURE)), store.months()[-1])

december, every_day = store.read(["2025-12"]), store.read(store.months())
write_that_fails("2025-12")  # a month the store has
check("a write that fails over a month the store has leaves that month as it was", store.read(["2025-12"]).equals(december))
try:
    whole_folder = len(pd.read_parquet(store.STORE))  # how lab/program.md tells the agent to read /data
except Exception as error:
    whole_folder = str(error).splitlines()[0][:80]
check("after that failed write, the whole folder still reads as one table, the way the agent reads /data",
      whole_folder == len(every_day), f"got {whole_folder!r}, expected {len(every_day)} days")


# --- make_data.py with other settings: 2023-01 to 2025-12 are still the same days -----------

made = run("scripts/make_data.py", "--start", "2021-07", "--filler", "2")
stored = store.months()
check("make_data.py --start 2021-07 --filler 2 makes a store that starts earlier, with two more columns",
      made.returncode == 0 and (stored[0], stored[-1]) == ("2021-07", "2025-12")
      and list(store.read(["2021-07"]).columns) == ["date", "temp_forecast", "load", "filler_001", "filler_002"],
      made.stderr[-300:] or (stored[:1], stored[-1:]))
days = store.read([month for month in stored if month >= "2023-01"])[["date", "temp_forecast", "load"]]
check("with those settings, 2023-01 to 2025-12 are still the old judge/full_data.csv, value for value",
      len(days) == 1096 and sha256_as_csv(days) == OLD_CSV_SHA256, f"{len(days)} days")

made = run("scripts/make_data.py", "--start", "2024-01")
stored = store.months()
check("make_data.py --start 2024-01 makes a store that starts later, with the same days from there on",
      made.returncode == 0 and stored[0] == "2024-01"
      and store.read(stored).equals(original_days[original_days.date >= "2024-01-01"].reset_index(drop=True)),
      made.stderr[-300:] or stored[:1])


shutil.rmtree(TMP)
print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
sys.exit(1 if failures else 0)
