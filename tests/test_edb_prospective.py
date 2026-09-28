import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from electrotrace.validation import match_peaks
from scripts import evaluate_frozen_model_edb_prospective as edb


class _Metadata:
    def __init__(self, protocol):
        frozen = protocol["frozen_model"]
        self.threshold = frozen["operating_threshold"]
        self._data = {
            "model_version": frozen["model_version"],
            "feature_schema_version": frozen["feature_schema_version"],
            "threshold": frozen["operating_threshold"],
            "target_recall": frozen["target_recall"],
            "n_training_candidates": frozen["n_training_candidates"],
            "n_positive_candidates": frozen["n_positive_candidates"],
            "n_negative_candidates": frozen["n_negative_candidates"],
            "random_seed": frozen["random_seed"],
            "n_estimators": frozen["n_estimators"],
            "calibration_candidates": frozen["calibration_candidates"],
            "calibration_method": frozen["calibration_method"],
        }

    def to_dict(self):
        return dict(self._data)


class _Model:
    fitted = True

    def __init__(self, protocol):
        self.metadata = _Metadata(protocol)


class _Audit:
    def __init__(self, record):
        self.record = record

    def to_dict(self):
        return {"record": self.record, "action": "kept_unchanged"}


def test_preregistered_protocol_hash_is_locked():
    assert edb.sha256_file(edb.PROTOCOL_PATH) == edb.EXPECTED_PROTOCOL_SHA256
    protocol = edb.load_locked_protocol()
    assert protocol["dataset"]["expected_record_count"] == 90
    assert len(protocol["dataset"]["records"]) == 90
    assert protocol["detector"]["channel"] == 0
    assert protocol["detector"]["expected_v2_gate_confidence"] == 0.0
    assert protocol["detector"]["expected_width_override_confidence"] == 0.0


def _write_dataset(tmp_path, records):
    (tmp_path / "RECORDS").write_text("\n".join(records) + "\n", encoding="utf-8")
    for record in records:
        for suffix in edb.REQUIRED_SUFFIXES:
            (tmp_path / f"{record}{suffix}").write_bytes(
                f"{record}{suffix}".encode("ascii")
            )


def test_dataset_verifier_requires_exact_ordered_90_record_cohort(tmp_path):
    protocol = edb.load_locked_protocol()
    records = protocol["dataset"]["records"]
    _write_dataset(tmp_path, records)
    hashes = edb.verify_dataset(tmp_path, records)
    assert len(hashes) == 1 + 90 * 3

    reversed_records = list(reversed(records))
    (tmp_path / "RECORDS").write_text(
        "\n".join(reversed_records) + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="does not exactly match"):
        edb.verify_dataset(tmp_path, records)


def test_dataset_verifier_aborts_on_missing_core_file(tmp_path):
    protocol = edb.load_locked_protocol()
    records = protocol["dataset"]["records"]
    _write_dataset(tmp_path, records)
    (tmp_path / f"{records[-1]}.atr").unlink()
    with pytest.raises(SystemExit, match="Incomplete EDB core files"):
        edb.verify_dataset(tmp_path, records)


def test_semantic_model_identity_is_locked():
    protocol = edb.load_locked_protocol()
    edb.verify_model(_Model(protocol), protocol)

    bad = _Model(protocol)
    bad.metadata._data["n_estimators"] = 999
    with pytest.raises(SystemExit, match="n_estimators"):
        edb.verify_model(bad, protocol)


def _install_fake_run(monkeypatch, tmp_path, *, fail_record=None):
    protocol = edb.load_locked_protocol()
    records = protocol["dataset"]["records"]
    model = _Model(protocol)

    model_path = tmp_path / "model.skops"
    model_path.write_bytes(b"safe-model")
    (tmp_path / "model.skops.json").write_text("{}", encoding="utf-8")
    derivation_path = tmp_path / "derive.json"
    derivation_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(edb, "verify_dataset", lambda root, expected: {"RECORDS": "0" * 64})
    monkeypatch.setattr(
        edb,
        "_load_polarity_threshold_report",
        lambda path: (
            {
                "schema": protocol["detector"]["polarity_threshold_report_schema"],
                "git_head": "a" * 40,
                "recommended_thresholds": {
                    "v2_gate_confidence": 0.0,
                    "width_override_confidence": 0.0,
                },
                "implementation_hashes": {"x": "y"},
            },
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        edb.CandidateSuppressor, "load", staticmethod(lambda path: model)
    )

    reference = np.array([100, 300, 500], dtype=int)

    def load_record(base, **kwargs):
        record = str(base).split("/")[-1]
        if fail_record is not None and record == fail_record:
            raise ValueError("synthetic record failure")
        return SimpleNamespace(
            record=record,
            fs_hz=250.0,
            signal=np.zeros(1000),
            reference=reference,
            audit=_Audit(record),
        )

    monkeypatch.setattr(edb, "load_annotated_record", load_record)
    monkeypatch.setattr(
        edb,
        "select_signal_polarity",
        lambda *args, **kwargs: SimpleNamespace(polarity="positive", confidence=1.0),
    )
    monkeypatch.setattr(
        edb,
        "detect_r_peaks_two_stage",
        lambda *args, **kwargs: (reference.copy(), np.ones(reference.size)),
    )
    monkeypatch.setattr(edb, "match_peaks", match_peaks)

    out = tmp_path / "first.json"
    monkeypatch.setattr(edb, "OUTPUT_PATH", out)
    return model_path, derivation_path, out, records


def test_full_fake_run_scores_all_90_and_marks_dataset_exposed(monkeypatch, tmp_path):
    model_path, derivation_path, out, _ = _install_fake_run(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_edb_prospective.py",
            "--edb-dir",
            str(tmp_path),
            "--model",
            str(model_path),
            "--polarity-threshold-report",
            str(derivation_path),
            "--confirm-first-scored-run",
        ],
    )
    assert edb.main() == 0
    report = json.loads(out.read_text())
    assert report["evidence_status"] == "prospective_external_evaluation_first_run"
    assert report["evaluation_integrity"]["records_scored"] == 90
    assert report["evaluation_integrity"]["records_skipped"] == 0
    assert len(report["record_results"]) == 90
    assert report["summary"]["f1"] == pytest.approx(1.0)
    assert report["selected_polarity_counts"] == {"positive": 90}
    assert report["dataset_exposure_after_this_run"].startswith("exposed")


def test_record_failure_aborts_without_partial_primary_artifact(monkeypatch, tmp_path):
    protocol = edb.load_locked_protocol()
    fail_record = protocol["dataset"]["records"][4]
    model_path, derivation_path, out, _ = _install_fake_run(
        monkeypatch, tmp_path, fail_record=fail_record
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_edb_prospective.py",
            "--edb-dir",
            str(tmp_path),
            "--model",
            str(model_path),
            "--polarity-threshold-report",
            str(derivation_path),
            "--confirm-first-scored-run",
        ],
    )
    with pytest.raises(SystemExit, match="no partial primary result is valid"):
        edb.main()
    assert not out.exists()


def test_runner_requires_explicit_first_run_acknowledgement(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_edb_prospective.py",
            "--edb-dir",
            str(tmp_path),
            "--model",
            str(tmp_path / "model.skops"),
            "--polarity-threshold-report",
            str(tmp_path / "derive.json"),
        ],
    )
    with pytest.raises(SystemExit, match="--confirm-first-scored-run"):
        edb.main()
