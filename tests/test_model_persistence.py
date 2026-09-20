import numpy as np
import pytest

from electrotrace.candidate_suppressor import CandidateSuppressor


skops = pytest.importorskip("skops")


def test_legacy_pickle_is_rejected_by_default(tmp_path):
    path = tmp_path / "legacy.pkl"
    path.write_bytes(b"not a real pickle")
    with pytest.raises(ValueError, match="Refusing legacy pickle"):
        CandidateSuppressor.load(path)


def test_model_persistence_requires_skops_extension():
    X = np.vstack([np.ones((8, 4)), -np.ones((8, 4))])
    y = np.array([1] * 8 + [0] * 8)
    model = CandidateSuppressor().fit(X, y, target_recall=0.9, n_estimators=10)
    with pytest.raises(ValueError, match="must end in .skops"):
        model.save("model.pkl")


def test_skops_roundtrip_preserves_predictions_and_metadata(tmp_path):
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(1, 0.1, size=(20, 6)), rng.normal(-1, 0.1, size=(20, 6))])
    y = np.array([1] * 20 + [0] * 20)
    model = CandidateSuppressor().fit(X, y, target_recall=0.9, n_estimators=15)
    model.feature_names = [f"f{i}" for i in range(X.shape[1])]

    path = tmp_path / "model.skops"
    model.save(path)
    loaded = CandidateSuppressor.load(path)

    assert loaded.metadata.feature_schema_version == model.metadata.feature_schema_version
    assert loaded.metadata.sklearn_version == model.metadata.sklearn_version
    assert loaded.feature_names == model.feature_names
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))


def test_skops_roundtrip_rejects_payload_hash_mismatch(tmp_path):
    X = np.vstack([np.ones((8, 4)), -np.ones((8, 4))])
    y = np.array([1] * 8 + [0] * 8)
    model = CandidateSuppressor().fit(X, y, target_recall=0.9, n_estimators=8)
    path = tmp_path / "model.skops"
    model.save(path)

    payload = path.with_suffix(".skops.json")
    text = payload.read_text(encoding="utf-8")
    data = __import__("json").loads(text)
    data["model_sha256"] = "0" * 64
    payload.write_text(__import__("json").dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        CandidateSuppressor.load(path)
