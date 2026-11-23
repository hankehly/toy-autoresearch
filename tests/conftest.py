"""What the test files share. pytest loads this file by itself, before any test file.

Run the tests from the repo root (Docker must be running; no login token is needed):

    uv run pytest                        # every test (about 3 minutes)
    uv run pytest tests/test_store.py    # one file
    uv run pytest -k proxy               # the tests with "proxy" in their names
    uv run pytest -v                     # one line per test, by name

No test touches data/ or as_of.txt, and none changes lab/, history/ or the ledgers. Each
test file gets an empty data folder of its own (the data_folder fixture below), so a fresh
checkout needs no `make_data.py` first, and an as-of file of its own that says 2025-12 (the
as_of_file fixture), so the scores a test expects stay right after the real as-of date has
advanced. GitHub runs the same command on every pull request (.github/workflows/tests.yml).

A fixture is a function that prepares something a test needs. A test asks for one by naming
it as a parameter. "module" scope means it runs once for a test file, "session" scope once
for the whole run, and `autouse` means every test gets it without asking.
"""
import importlib
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import judge
import loop
import store

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def images() -> None:
    """Build the three Docker images before the first test, so the tests check the
    sandboxes as the current files describe them. Quick when nothing has changed."""
    loop.build()


@pytest.fixture(scope="module", autouse=True)
def data_folder(tmp_path_factory) -> Iterator[Path]:
    """Give a test file an empty data folder of its own, in place of data/.

    TOY_DATA points store.py at the folder. The scripts and the judge that a test starts
    as separate programs follow TOY_DATA too, because they inherit it.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        pytest's own fixture, which makes temporary folders.

    Yields
    ------
    Path
        The folder. It is empty: there is no store until the test file makes one.
    """
    folder = tmp_path_factory.mktemp("data")
    with pytest.MonkeyPatch.context() as patch:  # undoes its changes when the test file is done
        patch.setenv("TOY_DATA", str(folder))
        importlib.reload(store)  # store.py reads TOY_DATA once, when it is loaded, so load it again
        yield folder
    importlib.reload(store)  # TOY_DATA is as it was before, so the store is data/ again
    shutil.rmtree(folder)


@pytest.fixture(scope="module", autouse=True)
def as_of_file(tmp_path_factory) -> Iterator[Path]:
    """Give a test file an as-of file of its own, in place of as_of.txt. It says 2025-12,
    as as_of.txt does on a fresh checkout.

    TOY_AS_OF points the judge and the loop at the file. The scripts and the judge that a
    test starts as separate programs follow TOY_AS_OF too, because they inherit it.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        pytest's own fixture, which makes temporary folders.

    Yields
    ------
    Path
        The file. A test that advances the as-of date changes this file, never as_of.txt.
    """
    path = tmp_path_factory.mktemp("as-of") / "as_of.txt"
    path.write_text("2025-12\n")
    with pytest.MonkeyPatch.context() as patch:  # undoes its changes when the test file is done
        patch.setenv("TOY_AS_OF", str(path))
        patch.setattr(loop, "AS_OF", path)   # loop.py and judge.py read TOY_AS_OF once, when they are loaded,
        patch.setattr(judge, "AS_OF", path)  # so point the copies already loaded here at the file too
        yield path


@pytest.fixture(scope="module")
def toy_store(data_folder) -> None:
    """Fill the test file's data folder with the toy's data, as a person does once after a
    fresh checkout. The store then holds 2023-01 to 2025-12.

    Parameters
    ----------
    data_folder : Path
        The test file's data folder, which TOY_DATA names.
    """
    subprocess.run([sys.executable, "scripts/make_data.py"], cwd=ROOT, check=True, capture_output=True)
