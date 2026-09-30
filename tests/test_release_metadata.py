import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    in_project = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "[project]":
            in_project = True
            continue
        if in_project and line.startswith("["):
            break
        if in_project and line.startswith("version ="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise AssertionError("project version not found")


def _runtime_version() -> str:
    text = (ROOT / "src" / "electrotrace" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*[\"\']([^\"\']+)', text, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def _citation_version() -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*[\"']?([^\"'\s]+)", text, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def _zenodo_version() -> str:
    data = json.loads((ROOT / "zenodo.json").read_text(encoding="utf-8"))
    return str(data["version"])


def test_release_version_metadata_is_synchronized():
    expected = _project_version()
    assert expected == "1.9.0"
    assert _runtime_version() == expected
    assert _citation_version() == expected
    assert _zenodo_version() == expected
