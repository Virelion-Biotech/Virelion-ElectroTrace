import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import app


def test_project_path_traversal_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ELECTROTRACE_PROJECT_ROOT", str(tmp_path / "projects"))
    # Project root is created at import time, so exercise the validation helper through a direct request.
    client = app.test_client()
    response = client.get("/api/project?name=../../escape")
    assert response.status_code == 400


def test_invalid_recording_id_rejected():
    client = app.test_client()
    response = client.get("/api/recording/not-a-valid-id/window?start=0&stop=10")
    assert response.status_code == 400
