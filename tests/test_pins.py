"""Shows that the Mac and both sandboxes install the same numpy, pandas and pyarrow.

    uv run pytest tests/test_pins.py

The versions are written in three files: pyproject.toml (the Mac, where the judge and the
loop run) and the Dockerfiles of the agent sandbox and the model sandbox. A model must score
the same wherever it runs, so the three files have to name the same versions. Before this
test, only a comment in each file said so.

It reads the three files and nothing else, so it takes under a second. Like every test, it
first waits for conftest.py to build the Docker images, which is quick when nothing changed.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PINNED = ("numpy", "pandas", "pyarrow")  # the packages that do a model's arithmetic
SANDBOXES = ("agent-sandbox", "model-sandbox")


def pins(text) -> dict[str, str]:
    """The exact versions a text pins: `{"numpy": "2.2.3"}` for `numpy==2.2.3`."""
    found = re.findall(r"\b(numpy|pandas|pyarrow)==([0-9][0-9A-Za-z.]*)", text)
    return dict(found)


@pytest.fixture(scope="module")
def on_the_mac() -> dict[str, str]:
    """The versions pyproject.toml pins."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return pins("\n".join(project["dependencies"]))


def test_pyproject_toml_pins_numpy_pandas_and_pyarrow_to_exact_versions(on_the_mac):
    assert sorted(on_the_mac) == sorted(PINNED)


@pytest.mark.parametrize("sandbox", SANDBOXES)
def test_the_sandbox_installs_the_versions_pyproject_toml_pins(sandbox, on_the_mac):
    dockerfile = (ROOT / "docker" / sandbox / "Dockerfile").read_text()
    assert pins(dockerfile) == on_the_mac


def test_the_search_itself_reads_a_pin_and_skips_a_range():
    assert pins("pip install numpy==2.2.3 pandas==2.2.3") == {"numpy": "2.2.3", "pandas": "2.2.3"}
    assert pins('"pyarrow>=26.0.1", "scipy==1.15.2"') == {}
