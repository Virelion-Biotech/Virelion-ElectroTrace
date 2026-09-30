import json
import re
import tomllib
from pathlib import Path

import electrotrace


ROOT = Path(__file__).resolve().parents[1]


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _citation_version() -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*["\']?([^"\'\s]+)', text, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def _zenodo_version() -> str:
    data = json.loads((ROOT / "zenodo.json").read_text(encoding="utf-8"))
    return str(data["version"])


def test_release_version_metadata_is_synchronized():
    expected = _project_version()
    assert expected == "1.9.0"
    assert electrotrace.__version__ == expected
    assert _citation_version() == expected
    assert _zenodo_version() == expected
