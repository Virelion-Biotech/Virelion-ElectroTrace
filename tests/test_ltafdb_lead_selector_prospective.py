import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from electrotrace.validation import DetectionMetrics, RecordValidation
from scripts import evaluate_ltafdb_lead_selector_prospective as lta


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


def test_protocol_is_git_blob_locked_and_has_84_records():
    assert lta.git_blob_sha(lta.PROTOCOL_PATH) == lta.EXPECTED_PROTOCOL_GIT_BLOB_SHA
    protocol = lta.load_locked_protocol()
    assert protocol["dataset"]["expected_record_count"] == 84
    assert len(protocol["dataset"]["records"]) == 84
    assert protocol["dataset"]["window"]["stop_sample_exclusive"] == 230400
    assert protocol["lead_selector"]["primary_retained_probability_p50_floor"] == pytest.approx(0.995)


def _write_dataset(tmp_path, records):
    (tmp_path / "RECORDS").write_text(" ".join(records) + "\n", encoding="utf-8")
    for record in records:
        for suffix in lta.REQUIRED_SUFFIXES:
            (tmp_path / f"{record}{suffix}").write_bytes(f"{record}{suffix}".encode())


def test_dataset_verifier_requires_exact_cohort(tmp_path):
    records = lta.load_locked_protocol()["dataset"]["records"]
    _write_dataset(tmp_path, records)
    hashes = lta.verify_dataset(tmp_path, records)
    assert len(hashes) == 1 + 84 * 3
    (tmp_path / "RECORDS").write_text(" ".join(reversed(records)) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not exactly match"):
        lta.verify_dataset(tmp_path, records)


def test_retained_probability_p50_is_fail_closed_and_has_frozen_empty_value():
    assert lta.retained_probability_p50(np.array([0.2, 0.8]), empty_value=0.0) == pytest.approx(0.5)
    assert lta.retained_probability_p50(np.array([]), empty_value=0.0) == 0.0
    with pytest.raises(ValueError):
        lta.retained_probability_p50(np.array([np.nan]), empty_value=0.0)


def test_evaluate_record_selects_lead_before_loading_annotations(monkeypatch, tmp_path):
    protocol = lta.load_locked_protocol()
    model = _Model(protocol)
    events = []
    signal = np.zeros((230400, 2), dtype=float)
    monkeypatch.setattr(
        lta.wfdb,
        "rdrecord",
        lambda *args, **kwargs: SimpleNamespace(
            fs=128.0, p_signal=signal, sig_name=["lead0", "lead1"]
        ),
    )
    calls = {"n": 0}

    def fake_detect(*args, **kwargs):
        calls["n"] += 1
        events.append("detect" + str(calls["n"]))
        probs = np.array([0.90, 0.90, 0.90]) if calls["n"] == 1 else np.array([0.999, 0.999, 0.999])
        return np.array([100, 300, 500], dtype=int), probs

    monkeypatch.setattr(lta, "detect_r_peaks_two_stage", fake_detect)

    def fake_choose(p0, p1, *, primary_floor):
        events.append("choose")
        assert p0 == pytest.approx(0.90)
        assert p1 == pytest.approx(0.999)
        assert primary_floor == pytest.approx(0.995)
        return 1

    monkeypatch.setattr(lta, "choose_two_lead_channel", fake_choose)

    def fake_rdann(*args, **kwargs):
        events.append("annotation")
        return SimpleNamespace(
            sample=np.array([100, 300, 500], dtype=int),
            symbol=["N", "N", "N"],
        )

    monkeypatch.setattr(lta.wfdb, "rdann", fake_rdann)
    result, payload, audit = lta.evaluate_record(
        tmp_path,
        "00",
        protocol=protocol,
        model=model,
        v2_gate=0.0,
        width_gate=0.0,
    )
    assert events.index("choose") < events.index("annotation")
    assert payload["selected_channel"] == 1
    assert result.metrics.f1 == pytest.approx(1.0)
    assert audit["lead_selected_before_annotation_load"] is True


def _perfect_result(record):
    metrics = DetectionMetrics(
        reference_count=3, detected_count=3, true_positive=3, false_positive=0, false_negative=0,
        sensitivity=1.0, positive_predictive_value=1.0, f1=1.0,
        mean_timing_error_ms=0.0, median_timing_error_ms=0.0,
        timing_error_sd_ms=0.0, mean_absolute_timing_error_ms=0.0,
        median_absolute_timing_error_ms=0.0, p95_absolute_timing_error_ms=0.0,
        max_absolute_timing_error_ms=0.0,
    )
    result = RecordValidation(record=record, fs_hz=128.0, metrics=metrics)
    payload = result.to_dict()
    payload.update({
        "selected_channel": 0,
        "signal_names": ["a", "b"],
        "channel0_retained_probability_p50": 0.999,
        "channel1_retained_probability_p50": 0.998,
        "channel0_detected_count": 3,
        "channel1_detected_count": 3,
        "reference_annotation_count": 3,
    })
    return result, payload, {"record": record, "lead_selected_before_annotation_load": True, "selected_channel": 0}


def _install_fake_main(monkeypatch, tmp_path, *, fail_record=None):
    protocol = lta.load_locked_protocol()
    model = _Model(protocol)
    model_path = tmp_path / "model.skops"
    model_path.write_bytes(b"model")
    (tmp_path / "model.skops.json").write_text("{}", encoding="utf-8")
    derivation = tmp_path / "derive.json"
    derivation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(lta, "verify_dataset", lambda root, records: {"RECORDS": "0" * 64})
    monkeypatch.setattr(
        lta,
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
    monkeypatch.setattr(lta.CandidateSuppressor, "load", staticmethod(lambda path: model))

    def fake_eval(root, record, **kwargs):
        if record == fail_record:
            raise ValueError("synthetic failure")
        return _perfect_result(record)

    monkeypatch.setattr(lta, "evaluate_record", fake_eval)
    out = tmp_path / "out.json"
    monkeypatch.setattr(lta, "OUTPUT_PATH", out)
    return model_path, derivation, out, protocol


def test_full_fake_first_run_scores_all_84(monkeypatch, tmp_path):
    model, derivation, out, _ = _install_fake_main(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ltafdb_lead_selector_prospective.py",
        "--ltafdb-dir", str(tmp_path),
        "--model", str(model),
        "--polarity-threshold-report", str(derivation),
        "--confirm-first-scored-run",
    ])
    assert lta.main() == 0
    report = json.loads(out.read_text())
    assert report["evidence_status"] == "prospective_external_lead_selector_evaluation_first_run"
    assert report["evaluation_integrity"]["records_scored"] == 84
    assert report["evaluation_integrity"]["lead_selection_used_reference_annotations"] is False
    assert report["summary"]["f1"] == pytest.approx(1.0)
    assert report["selected_channel_counts"] == {"0": 84}


def test_failure_aborts_without_partial_metric_exposure(monkeypatch, tmp_path, capsys):
    protocol = lta.load_locked_protocol()
    fail_record = protocol["dataset"]["records"][3]
    model, derivation, out, _ = _install_fake_main(monkeypatch, tmp_path, fail_record=fail_record)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ltafdb_lead_selector_prospective.py",
        "--ltafdb-dir", str(tmp_path),
        "--model", str(model),
        "--polarity-threshold-report", str(derivation),
        "--confirm-first-scored-run",
    ])
    with pytest.raises(SystemExit, match="no partial primary result is valid"):
        lta.main()
    captured = capsys.readouterr()
    assert "f1=" not in captured.out.lower()
    assert "ppv=" not in captured.out.lower()
    assert "sens=" not in captured.out.lower()
    assert not out.exists()
