import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from electrotrace.lead_quality import LeadQuality
from electrotrace.validation import RecordValidation, match_peaks
from scripts import evaluate_svdb_selector_v2_prospective as sv


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


def _quality(p50, qrs, retention):
    return LeadQuality(
        retained_probability_p50=p50,
        retained_qrs_band_fraction=qrs,
        retention_fraction=retention,
        stage1_candidate_count=4,
        retained_count=3,
        selected_polarity="positive",
        polarity_confidence=1.0,
    )


def test_protocol_is_locked_and_has_exact_78_record_manifest():
    assert sv.git_blob_sha(sv.PROTOCOL_PATH) == sv.EXPECTED_PROTOCOL_GIT_BLOB_SHA
    protocol = sv.load_locked_protocol()
    assert protocol["dataset"]["expected_record_count"] == 78
    assert len(protocol["dataset"]["records"]) == 78
    assert protocol["dataset"]["records"][0] == "800"
    assert protocol["dataset"]["records"][-1] == "894"
    assert protocol["lead_selector"]["version"] == "edb-ltafdb-informed-quality-consensus-v2"


def _write_fake_dataset(tmp_path, records):
    (tmp_path / "RECORDS").write_text(" ".join(records) + "\n", encoding="utf-8")
    for record in records:
        for suffix in sv.REQUIRED_SUFFIXES:
            (tmp_path / f"{record}{suffix}").write_bytes(f"{record}{suffix}".encode())


def test_dataset_verifier_checks_manifest_and_header_geometry(monkeypatch, tmp_path):
    protocol = sv.load_locked_protocol()
    records = protocol["dataset"]["records"]
    _write_fake_dataset(tmp_path, records)
    monkeypatch.setattr(
        sv.wfdb,
        "rdheader",
        lambda path: SimpleNamespace(n_sig=2, fs=128.0, sig_len=230400),
    )
    hashes = sv.verify_dataset(tmp_path, protocol)
    assert len(hashes) == 1 + 78 * 3

    monkeypatch.setattr(
        sv.wfdb,
        "rdheader",
        lambda path: SimpleNamespace(n_sig=2, fs=250.0, sig_len=230400),
    )
    with pytest.raises(SystemExit, match="expected 128.0 Hz"):
        sv.verify_dataset(tmp_path, protocol)


def test_reference_filter_allows_nonbeat_endpoint_marker_but_not_endpoint_beat():
    symbols = frozenset({"N"})
    annotation = SimpleNamespace(
        sample=np.array([100, 230400], dtype=int),
        symbol=["N", "+"],
    )
    reference, audit = sv._reference_from_annotation(annotation, symbols, 230400)
    assert reference.tolist() == [100]
    assert audit["out_of_range_nonbeat_annotations_dropped"] == 1

    bad = SimpleNamespace(
        sample=np.array([100, 230400], dtype=int),
        symbol=["N", "N"],
    )
    with pytest.raises(ValueError, match="reference beat samples fall outside"):
        sv._reference_from_annotation(bad, symbols, 230400)

def test_annotation_window_excludes_nominal_endpoint(monkeypatch, tmp_path):
    protocol = sv.load_locked_protocol()
    model = _Model(protocol)
    signal = np.zeros((230400, 2), dtype=float)

    monkeypatch.setattr(
        sv.wfdb,
        "rdrecord",
        lambda *args, **kwargs: SimpleNamespace(
            fs=128.0, p_signal=signal, sig_name=["ECG1", "ECG2"]
        ),
    )
    monkeypatch.setattr(
        sv,
        "compute_lead_quality",
        lambda *args, **kwargs: (
            np.array([100, 300, 500]),
            np.array([0.9, 0.95, 0.99]),
            _quality(0.99, 0.5, 0.6),
        ),
    )
    monkeypatch.setattr(sv, "choose_two_lead_channel_v2", lambda *args, **kwargs: 0)

    def fake_rdann(*args, **kwargs):
        assert kwargs == {"sampfrom": 0, "sampto": 230399}
        return SimpleNamespace(
            sample=np.array([100, 300, 500]),
            symbol=["N", "N", "N"],
        )

    monkeypatch.setattr(sv.wfdb, "rdann", fake_rdann)
    result, _, _ = sv.evaluate_record(
        tmp_path,
        "860",
        protocol=protocol,
        model=model,
        v2_gate=0.0,
        width_gate=0.0,
    )
    assert result.metrics.f1 == pytest.approx(1.0)


def test_evaluate_record_selects_v2_before_annotation_load(monkeypatch, tmp_path):
    protocol = sv.load_locked_protocol()
    model = _Model(protocol)
    events = []
    signal = np.zeros((230400, 2), dtype=float)
    monkeypatch.setattr(
        sv.wfdb,
        "rdrecord",
        lambda *args, **kwargs: SimpleNamespace(
            fs=128.0, p_signal=signal, sig_name=["ECG1", "ECG2"]
        ),
    )
    calls = {"n": 0}

    def fake_quality(*args, **kwargs):
        calls["n"] += 1
        events.append("quality" + str(calls["n"]))
        if calls["n"] == 1:
            return np.array([100, 300, 500]), np.array([0.8, 0.9, 0.9]), _quality(0.90, 0.2, 0.4)
        return np.array([100, 300, 500]), np.array([0.9, 0.95, 0.99]), _quality(0.95, 0.5, 0.6)

    monkeypatch.setattr(sv, "compute_lead_quality", fake_quality)

    def fake_choose(*args, **kwargs):
        events.append("choose")
        return 1

    monkeypatch.setattr(sv, "choose_two_lead_channel_v2", fake_choose)

    def fake_rdann(*args, **kwargs):
        events.append("annotation")
        assert kwargs["sampfrom"] == 0
        assert kwargs["sampto"] == 230399
        return SimpleNamespace(
            sample=np.array([100, 300, 500]),
            symbol=["N", "N", "N"],
        )

    monkeypatch.setattr(sv.wfdb, "rdann", fake_rdann)
    result, payload, audit = sv.evaluate_record(
        tmp_path,
        "800",
        protocol=protocol,
        model=model,
        v2_gate=0.0,
        width_gate=0.0,
    )
    assert events.index("choose") < events.index("annotation")
    assert payload["selected_channel"] == 1
    assert payload["channel1_quality"]["retained_qrs_band_fraction"] == pytest.approx(0.5)
    assert result.metrics.f1 == pytest.approx(1.0)
    assert audit["lead_selected_before_annotation_load"] is True


def _perfect_result(record):
    metrics = match_peaks([100, 300, 500], [100, 300, 500], 128.0, tolerance_ms=75.0)
    result = RecordValidation(record=record, fs_hz=128.0, metrics=metrics)
    payload = result.to_dict()
    payload.update({
        "selected_channel": 0,
        "signal_names": ["ECG1", "ECG2"],
        "channel0_quality": _quality(0.999, 0.6, 0.7).to_dict(),
        "channel1_quality": _quality(0.8, 0.5, 0.6).to_dict(),
        "reference_annotation_count": 3,
    })
    return result, payload, {
        "record": record,
        "lead_selected_before_annotation_load": True,
        "selected_channel": 0,
    }


def _install_fake_main(monkeypatch, tmp_path, *, fail_record=None):
    protocol = sv.load_locked_protocol()
    model = _Model(protocol)
    model_path = tmp_path / "model.skops"
    model_path.write_bytes(b"model")
    (tmp_path / "model.skops.json").write_text("{}", encoding="utf-8")
    derivation = tmp_path / "derive.json"
    derivation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sv, "verify_dataset", lambda root, p: {"RECORDS": "0" * 64})
    monkeypatch.setattr(
        sv,
        "_load_polarity_threshold_report",
        lambda path: (
            {
                "schema": protocol["detector"]["polarity_threshold_report_schema"],
                "git_head": "a" * 40,
                "recommended_thresholds": {
                    "v2_gate_confidence": 0.0,
                    "width_override_confidence": 0.0,
                },
                "implementation_hashes": {},
            },
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr(sv.CandidateSuppressor, "load", staticmethod(lambda path: model))

    def fake_eval(root, record, **kwargs):
        if record == fail_record:
            raise ValueError("synthetic failure")
        return _perfect_result(record)

    monkeypatch.setattr(sv, "evaluate_record", fake_eval)
    out = tmp_path / "out.json"
    monkeypatch.setattr(sv, "OUTPUT_PATH", out)
    return model_path, derivation, out, protocol


def test_full_fake_run_scores_all_78(monkeypatch, tmp_path):
    model, derivation, out, _ = _install_fake_main(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_svdb_selector_v2_prospective.py",
        "--svdb-dir", str(tmp_path),
        "--model", str(model),
        "--polarity-threshold-report", str(derivation),
        "--confirm-first-scored-run",
    ])
    assert sv.main() == 0
    report = json.loads(out.read_text())
    assert report["evaluation_integrity"]["records_scored"] == 78
    assert report["evaluation_integrity"]["lead_selection_used_reference_annotations"] is False
    assert report["summary"]["f1"] == pytest.approx(1.0)
    assert report["selected_channel_counts"] == {"0": 78}


def test_record_failure_aborts_without_partial_outcome_metrics(monkeypatch, tmp_path, capsys):
    protocol = sv.load_locked_protocol()
    fail_record = protocol["dataset"]["records"][4]
    model, derivation, out, _ = _install_fake_main(
        monkeypatch, tmp_path, fail_record=fail_record
    )
    monkeypatch.setattr(sys, "argv", [
        "evaluate_svdb_selector_v2_prospective.py",
        "--svdb-dir", str(tmp_path),
        "--model", str(model),
        "--polarity-threshold-report", str(derivation),
        "--confirm-first-scored-run",
    ])
    with pytest.raises(SystemExit, match="no partial primary result is valid"):
        sv.main()
    captured = capsys.readouterr()
    assert "f1=" not in captured.out.lower()
    assert "ppv=" not in captured.out.lower()
    assert "sens=" not in captured.out.lower()
    assert not out.exists()
