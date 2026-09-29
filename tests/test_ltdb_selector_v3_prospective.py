import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from electrotrace.lead_quality import LeadQuality
from electrotrace.validation import RecordValidation, match_peaks
from scripts import evaluate_ltdb_selector_v3_prospective as ltd


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


def _quality(p50, qrs, retention, *, stage1=4, retained=3):
    return LeadQuality(
        retained_probability_p50=p50,
        retained_qrs_band_fraction=qrs,
        retention_fraction=retention,
        stage1_candidate_count=stage1,
        retained_count=retained,
        selected_polarity="positive",
        polarity_confidence=1.0,
    )


def test_protocol_is_locked_and_has_exact_7_record_manifest():
    assert ltd.git_blob_sha(ltd.PROTOCOL_PATH) == ltd.EXPECTED_PROTOCOL_GIT_BLOB_SHA
    protocol = ltd.load_locked_protocol()
    assert protocol["dataset"]["expected_record_count"] == 7
    assert protocol["dataset"]["records"] == [
        "14046", "14134", "14149", "14157", "14172", "14184", "15814"
    ]
    assert protocol["dataset"]["candidate_channels"] == [0, 1]
    assert protocol["dataset"]["expected_channel_counts"]["15814"] == 3
    assert (
        protocol["lead_selector"]["version"]
        == "edb-ltafdb-svdb-informed-starvation-rescue-v3"
    )


def _write_fake_dataset(tmp_path, records):
    (tmp_path / "RECORDS").write_text(" ".join(records) + "\n", encoding="utf-8")
    for record in records:
        for suffix in ltd.REQUIRED_SUFFIXES:
            (tmp_path / f"{record}{suffix}").write_bytes(f"{record}{suffix}".encode())


def test_dataset_verifier_checks_manifest_and_header_geometry(monkeypatch, tmp_path):
    protocol = ltd.load_locked_protocol()
    records = protocol["dataset"]["records"]
    _write_fake_dataset(tmp_path, records)

    def good_header(path):
        name = str(path).split("/")[-1]
        return SimpleNamespace(
            n_sig=3 if name == "15814" else 2,
            fs=128.0,
            sig_len=230400,
        )

    monkeypatch.setattr(ltd.wfdb, "rdheader", good_header)
    hashes = ltd.verify_dataset(tmp_path, protocol)
    assert len(hashes) == 1 + 7 * 3

    monkeypatch.setattr(
        ltd.wfdb,
        "rdheader",
        lambda path: SimpleNamespace(n_sig=2, fs=250.0, sig_len=230400),
    )
    with pytest.raises(SystemExit, match="expected 128.0 Hz"):
        ltd.verify_dataset(tmp_path, protocol)


def test_dataset_verifier_enforces_record_15814_three_channel_header(monkeypatch, tmp_path):
    protocol = ltd.load_locked_protocol()
    _write_fake_dataset(tmp_path, protocol["dataset"]["records"])

    def wrong_15814(path):
        return SimpleNamespace(n_sig=2, fs=128.0, sig_len=230400)

    monkeypatch.setattr(ltd.wfdb, "rdheader", wrong_15814)
    with pytest.raises(SystemExit, match="15814: expected 3 channels"):
        ltd.verify_dataset(tmp_path, protocol)


def test_reference_filter_allows_nonbeat_endpoint_marker_but_not_endpoint_beat():
    symbols = frozenset({"N"})
    annotation = SimpleNamespace(
        sample=np.array([100, 230400], dtype=int),
        symbol=["N", "+"],
    )
    reference, audit = ltd._reference_from_annotation(annotation, symbols, 230400)
    assert reference.tolist() == [100]
    assert audit["out_of_range_nonbeat_annotations_dropped"] == 1

    bad = SimpleNamespace(
        sample=np.array([100, 230400], dtype=int),
        symbol=["N", "N"],
    )
    with pytest.raises(ValueError, match="reference beat samples fall outside"):
        ltd._reference_from_annotation(bad, symbols, 230400)

def test_annotation_window_excludes_nominal_endpoint(monkeypatch, tmp_path):
    protocol = ltd.load_locked_protocol()
    model = _Model(protocol)
    signal = np.zeros((230400, 2), dtype=float)

    monkeypatch.setattr(
        ltd.wfdb,
        "rdrecord",
        lambda *args, **kwargs: SimpleNamespace(
            fs=128.0, p_signal=signal, sig_name=["ECG1", "ECG2"]
        ),
    )
    monkeypatch.setattr(
        ltd,
        "compute_lead_quality",
        lambda *args, **kwargs: (
            np.array([100, 300, 500]),
            np.array([0.9, 0.95, 0.99]),
            _quality(0.99, 0.5, 0.6),
        ),
    )
    monkeypatch.setattr(ltd, "choose_two_lead_channel_v3", lambda *args, **kwargs: 0)

    def fake_rdann(*args, **kwargs):
        assert kwargs == {"sampfrom": 0, "sampto": 230399}
        return SimpleNamespace(
            sample=np.array([100, 300, 500]),
            symbol=["N", "N", "N"],
        )

    monkeypatch.setattr(ltd.wfdb, "rdann", fake_rdann)
    result, _, _ = ltd.evaluate_record(
        tmp_path,
        "14046",
        protocol=protocol,
        model=model,
        v2_gate=0.0,
        width_gate=0.0,
    )
    assert result.metrics.f1 == pytest.approx(1.0)


def test_evaluate_record_selects_v3_before_annotation_load(monkeypatch, tmp_path):
    protocol = ltd.load_locked_protocol()
    model = _Model(protocol)
    events = []
    signal = np.zeros((230400, 2), dtype=float)

    def fake_rdrecord(*args, **kwargs):
        assert kwargs["sampfrom"] == 0
        assert kwargs["sampto"] == 230400
        assert kwargs["channels"] == [0, 1]
        return SimpleNamespace(
            fs=128.0, p_signal=signal, sig_name=["ECG1", "ECG2"]
        )

    monkeypatch.setattr(ltd.wfdb, "rdrecord", fake_rdrecord)
    calls = {"n": 0}

    def fake_quality(*args, **kwargs):
        calls["n"] += 1
        events.append("quality" + str(calls["n"]))
        if calls["n"] == 1:
            return (
                np.array([100] * 100),
                np.full(100, 0.80),
                _quality(0.80, 0.3, 0.10, stage1=1000, retained=100),
            )
        return (
            np.array([100, 300, 500]),
            np.array([0.95, 0.98, 0.99]),
            _quality(0.98, 0.6, 0.70, stage1=3000, retained=2100),
        )

    monkeypatch.setattr(ltd, "compute_lead_quality", fake_quality)

    def fake_choose(*args, **kwargs):
        events.append("choose")
        assert args[0] == pytest.approx(100 / 30)
        assert args[1] == pytest.approx(2100 / 30)
        assert args[2] == pytest.approx(0.10)
        assert args[3] == pytest.approx(0.70)
        return 1

    monkeypatch.setattr(ltd, "choose_two_lead_channel_v3", fake_choose)

    def fake_rdann(*args, **kwargs):
        events.append("annotation")
        assert kwargs == {"sampfrom": 0, "sampto": 230399}
        return SimpleNamespace(
            sample=np.array([100, 300, 500]),
            symbol=["N", "N", "N"],
        )

    monkeypatch.setattr(ltd.wfdb, "rdann", fake_rdann)
    result, payload, audit = ltd.evaluate_record(
        tmp_path,
        "14046",
        protocol=protocol,
        model=model,
        v2_gate=0.0,
        width_gate=0.0,
    )
    assert events.index("choose") < events.index("annotation")
    assert payload["selected_channel"] == 1
    assert payload["channel0_retained_rate_bpm"] == pytest.approx(100 / 30)
    assert payload["channel1_retained_rate_bpm"] == pytest.approx(2100 / 30)
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
        "channel0_retained_rate_bpm": 0.1,
        "channel1_retained_rate_bpm": 0.1,
        "reference_annotation_count": 3,
    })
    return result, payload, {
        "record": record,
        "lead_selected_before_annotation_load": True,
        "selected_channel": 0,
    }


def _install_fake_main(monkeypatch, tmp_path, *, fail_record=None):
    protocol = ltd.load_locked_protocol()
    model = _Model(protocol)
    model_path = tmp_path / "model.skops"
    model_path.write_bytes(b"model")
    (tmp_path / "model.skops.json").write_text("{}", encoding="utf-8")
    derivation = tmp_path / "derive.json"
    derivation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ltd, "verify_dataset", lambda root, p: {"RECORDS": "0" * 64})
    monkeypatch.setattr(
        ltd,
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
    monkeypatch.setattr(ltd.CandidateSuppressor, "load", staticmethod(lambda path: model))

    def fake_eval(root, record, **kwargs):
        if record == fail_record:
            raise ValueError("synthetic failure")
        return _perfect_result(record)

    monkeypatch.setattr(ltd, "evaluate_record", fake_eval)
    out = tmp_path / "out.json"
    monkeypatch.setattr(ltd, "OUTPUT_PATH", out)
    return model_path, derivation, out, protocol


def test_full_fake_run_scores_all_7(monkeypatch, tmp_path):
    model, derivation, out, _ = _install_fake_main(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ltdb_selector_v3_prospective.py",
        "--ltdb-dir", str(tmp_path),
        "--model", str(model),
        "--polarity-threshold-report", str(derivation),
        "--confirm-first-scored-run",
    ])
    assert ltd.main() == 0
    report = json.loads(out.read_text())
    assert report["evaluation_integrity"]["records_scored"] == 7
    assert report["evaluation_integrity"]["lead_selection_used_reference_annotations"] is False
    assert report["summary"]["f1"] == pytest.approx(1.0)
    assert report["selected_channel_counts"] == {"0": 7}


def test_record_failure_aborts_without_partial_outcome_metrics(monkeypatch, tmp_path, capsys):
    protocol = ltd.load_locked_protocol()
    fail_record = protocol["dataset"]["records"][4]
    model, derivation, out, _ = _install_fake_main(
        monkeypatch, tmp_path, fail_record=fail_record
    )
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ltdb_selector_v3_prospective.py",
        "--ltdb-dir", str(tmp_path),
        "--model", str(model),
        "--polarity-threshold-report", str(derivation),
        "--confirm-first-scored-run",
    ])
    with pytest.raises(SystemExit, match="no partial primary result is valid"):
        ltd.main()
    captured = capsys.readouterr()
    assert "f1=" not in captured.out.lower()
    assert "ppv=" not in captured.out.lower()
    assert "sens=" not in captured.out.lower()
    assert not out.exists()
