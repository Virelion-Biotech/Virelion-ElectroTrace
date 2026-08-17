from electrotrace.project_store import ProjectStore, RecordingRef
from electrotrace.security import ApiKeyMiddleware, validate_bind


def test_validate_bind_allows_localhost_without_key():
    validate_bind("127.0.0.1", None)
    validate_bind("::1", None)


def test_validate_bind_rejects_external_without_key():
    import pytest
    with pytest.raises(RuntimeError):
        validate_bind("0.0.0.0", None)


def test_api_key_middleware_blocks_missing_key():
    from werkzeug.test import Client
    from werkzeug.wrappers import Response
    client = Client(ApiKeyMiddleware(lambda environ, start_response: Response("ok")(environ, start_response), "secret"), Response)
    response = client.get("/api/recording")
    assert response.status_code == 401


def test_api_key_middleware_accepts_valid_bearer_token():
    from werkzeug.test import Client
    from werkzeug.wrappers import Response
    client = Client(ApiKeyMiddleware(lambda environ, start_response: Response("ok")(environ, start_response), "secret"), Response)
    response = client.get("/api/recording", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.data == b"ok"


def test_project_add_recording_is_idempotent(tmp_path):
    store = ProjectStore(tmp_path / "p")
    ref = RecordingRef(recording_id="r1", subject_id="s1")
    first = store.add_recording(ref)
    second = store.add_recording(ref)
    assert len(first.recordings) == 1
    assert len(second.recordings) == 1
    assert (tmp_path / "p" / ".project.lock").exists()
