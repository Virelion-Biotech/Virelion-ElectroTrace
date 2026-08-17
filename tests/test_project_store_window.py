from pathlib import Path

import pandas as pd

from electrotrace.project_store import ProjectStore


def test_csv_project_window_returns_requested_rows(tmp_path: Path):
    path = tmp_path / "recording.csv"
    pd.DataFrame({"time": range(1000), "Lead_II": range(1000)}).to_csv(path, index=False)

    store = ProjectStore(tmp_path / "project")
    window = store.window(path, 120, 130)

    assert list(window["time"]) == list(range(120, 130))
    assert list(window["Lead_II"]) == list(range(120, 130))


def test_csv_project_window_rejects_invalid_range(tmp_path: Path):
    path = tmp_path / "recording.csv"
    pd.DataFrame({"time": [0, 1, 2], "Lead_II": [1, 2, 3]}).to_csv(path, index=False)
    store = ProjectStore(tmp_path / "project")

    try:
        store.window(path, 2, 2)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "0 <= start < stop" in str(exc)
