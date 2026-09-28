import json
import sys
import types

import numpy as np
import pytest

from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features
from scripts import evaluate_frozen_model_mitdb as eval_script

FS = 360.0


def _peaks(n=20, step=300, start=200):
    return np.arange(start, start + n * step, step)


def _record(peaks):
    signal = np.zeros(int(peaks[-1] + 2000))
    for peak in peaks:
        signal[peak - 4 : peak + 5] += np.hanning(9) * 2.5
    return signal


def _tiny_model(signal, peaks):
    features, names = _candidate_features(signal, FS, peaks, np.ones(len(peaks)))
    model = CandidateSuppressor().fit(
        np.vstack([features, features + 0.01]),
        np.array([1] * len(peaks) + [0] * len(peaks)),
        target_recall=0.9,
        n_estimators=20,
    )
    model.feature_names = names
    return model


def _fake_wfdb(monkeypatch, records):
    fake = types.ModuleType("wfdb")

    class Obj:
        pass

    def rdrecord(base, channels, physical):
        signal, fs, _ = records[base.split("/")[-1]]
        obj = Obj()
        obj.fs = fs
        obj.p_signal = None
        obj.d_signal = signal.reshape(-1, 1)
        return obj

    def rdheader(base):
        signal, fs, _ = records[base.split("/")[-1]]
        obj = Obj()
        obj.fs = fs
        obj.sig_len = len(signal)
        return obj

    def rdann(base, extension):
        _, _, reference = records[base.split("/")[-1]]
        obj = Obj()
        obj.sample = np.asarray(reference)
        obj.symbol = ["N"] * len(reference)
        return obj

    fake.rdrecord = rdrecord
    fake.rdheader = rdheader
    fake.rdann = rdann
    monkeypatch.setitem(sys.modules, "wfdb", fake)


def _write_stub_record(tmp_path, name):
    for suffix in (".hea", ".dat", ".atr"):
        (tmp_path / f"{name}{suffix}").write_bytes(b"x")


def _derivation_report(path, *, v2=0.07, width=0.38, freeze_eligible=True, incart_used=False):
    report = {
        "schema": eval_script.DERIVATION_SCHEMA,
        "selection_status": "development_only_full_pool",
        "freeze_eligible": freeze_eligible,
        "git_head": eval_script.git_head(),
        "input_hashes": {
            f"R{i:02d}{suffix}": f"hash-{i}-{suffix}"
            for i in range(eval_script.EXPECTED_DEVELOPMENT_POOL_SIZE)
            for suffix in (".hea", ".dat", ".atr")
        },
        "protocol": {
            "development_pool_size": eval_script.EXPECTED_DEVELOPMENT_POOL_SIZE,
            "locked_heldout_labels_used": False,
            "incart_used": incart_used,
            "diagnostic_subset": False,
        },
        "recommended_thresholds": {
            "v2_gate_confidence": v2,
            "width_override_confidence": width,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _base_runtime(monkeypatch, tmp_path, record_name="105"):
    peaks = _peaks()
    signal = _record(peaks)
    model = _tiny_model(signal, peaks)
    _write_stub_record(tmp_path, record_name)
    _fake_wfdb(monkeypatch, {record_name: (signal, FS, peaks)})
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kwargs: model))
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")
    return model_file


def test_nondefault_v2_gate_requires_derivation_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(tmp_path / "model.skops"),
            "--width-override-confidence",
            "0.38",
            "--v2-gate-confidence",
            "0.07",
        ],
    )
    with pytest.raises(SystemExit, match="requires --polarity-threshold-report"):
        eval_script.main()


def test_derivation_report_must_match_current_code_revision(tmp_path):
    path = _derivation_report(tmp_path / "derive.json")
    report = json.loads(path.read_text())
    report["git_head"] = "0" * 40
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match current checkout"):
        eval_script._load_polarity_threshold_report(path)


def test_derivation_report_must_contain_complete_input_hash_set(tmp_path):
    path = _derivation_report(tmp_path / "derive.json")
    report = json.loads(path.read_text())
    report["input_hashes"].pop(next(iter(report["input_hashes"])))
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="complete 36-record core input hash set"):
        eval_script._load_polarity_threshold_report(path)


@pytest.mark.parametrize(
    ("freeze_eligible", "incart_used", "message"),
    [
        (False, False, "not freeze_eligible"),
        (True, True, "used INCART"),
    ],
)
def test_derivation_report_must_prove_evaluation_separation(
    tmp_path, freeze_eligible, incart_used, message
):
    path = _derivation_report(
        tmp_path / "derive.json",
        freeze_eligible=freeze_eligible,
        incart_used=incart_used,
    )
    with pytest.raises(SystemExit, match=message):
        eval_script._load_polarity_threshold_report(path)


def test_evaluator_threads_verified_report_gate_and_records_provenance(monkeypatch, tmp_path):
    model_file = _base_runtime(monkeypatch, tmp_path)
    derivation = _derivation_report(tmp_path / "derive.json", v2=0.07, width=0.38)

    seen = []
    real = eval_script.detect_r_peaks_two_stage

    def spy(*args, **kwargs):
        seen.append(
            (kwargs.get("width_override_confidence"), kwargs.get("v2_gate_confidence"))
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(eval_script, "detect_r_peaks_two_stage", spy)
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--records",
            "105",
            "--polarity-threshold-report",
            str(derivation),
            "--output",
            str(out),
        ],
    )
    assert eval_script.main() == 0
    report = json.loads(out.read_text())
    assert seen == [(0.38, 0.07)]
    assert report["protocol"]["width_override_confidence"] == pytest.approx(0.38)
    assert report["protocol"]["v2_gate_confidence"] == pytest.approx(0.07)
    assert report["protocol"]["v2_gate_confidence_is_historical_default"] is False
    assert report["protocol"]["polarity_threshold_derivation"]["sha256"]
    assert report["evaluation_integrity"]["threshold_derivation_report_verified"] is True
    assert report["evidence_status"] == "diagnostic_protocol_override"
    assert report["evaluation_integrity"]["protocol_override_reasons"] == ["locked_record_subset"]


def test_nondefault_width_gate_requires_derivation_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(tmp_path / "model.skops"),
            "--width-override-confidence",
            "0.05",
        ],
    )
    with pytest.raises(SystemExit, match="requires --polarity-threshold-report"):
        eval_script.main()


def test_cli_values_must_match_authoritative_derivation_report(monkeypatch, tmp_path):
    derivation = _derivation_report(tmp_path / "derive.json", v2=0.07, width=0.38)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(tmp_path / "model.skops"),
            "--width-override-confidence",
            "0.30",
            "--polarity-threshold-report",
            str(derivation),
        ],
    )
    with pytest.raises(SystemExit, match="disagrees"):
        eval_script.main()


def test_historical_default_without_report_is_allowed_but_labeled(monkeypatch, tmp_path):
    model_file = _base_runtime(monkeypatch, tmp_path)
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--records",
            "105",
            "--width-override-confidence",
            "0.38",
            "--output",
            str(out),
        ],
    )
    assert eval_script.main() == 0
    report = json.loads(out.read_text())
    assert report["protocol"]["v2_gate_confidence"] == pytest.approx(0.15)
    assert report["protocol"]["v2_gate_confidence_is_historical_default"] is True
    assert report["protocol"]["polarity_threshold_derivation"] is None
    assert report["evaluation_integrity"]["adaptive_polarity_historically_informed_by_locked_data"] is True


def test_full_locked_adaptive_run_is_labeled_legacy_not_prospective(monkeypatch, tmp_path):
    peaks = _peaks()
    signal = _record(peaks)
    model = _tiny_model(signal, peaks)
    records = {}
    for name in eval_script.LOCKED_HELDOUT_RECORDS:
        _write_stub_record(tmp_path, name)
        records[name] = (signal, FS, peaks)
    _fake_wfdb(monkeypatch, records)
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kwargs: model))
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")

    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--width-override-confidence",
            "0.38",
            "--output",
            str(out),
        ],
    )
    assert eval_script.main() == 0
    report = json.loads(out.read_text())
    assert report["evidence_status"] == "legacy_validation_non_regression"
    assert report["evaluation_integrity"]["full_locked_split"] is True
    assert report["evaluation_integrity"]["prospective_adaptive_validation"] is False


def test_full_locked_protocol_override_is_diagnostic(monkeypatch, tmp_path):
    peaks = _peaks()
    signal = _record(peaks)
    model = _tiny_model(signal, peaks)
    records = {}
    for name in eval_script.LOCKED_HELDOUT_RECORDS:
        _write_stub_record(tmp_path, name)
        records[name] = (signal, FS, peaks)
    _fake_wfdb(monkeypatch, records)
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kwargs: model))
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")

    out = tmp_path / "diagnostic.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--width-override-confidence",
            "0.38",
            "--polarity",
            "positive",
            "--output",
            str(out),
        ],
    )
    assert eval_script.main() == 0
    report = json.loads(out.read_text())
    assert report["evidence_status"] == "diagnostic_protocol_override"
    assert report["evaluation_integrity"]["protocol_override_reasons"] == [
        "polarity_override:positive"
    ]
    assert "do not promote" in report["evaluation_integrity"]["interpretation"]


def test_records_cannot_escape_locked_split(monkeypatch, tmp_path):
    model_file = _base_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--records",
            "115",
            "--width-override-confidence",
            "0.38",
        ],
    )
    with pytest.raises(SystemExit, match="locked MIT-BIH split"):
        eval_script.main()


@pytest.mark.parametrize("value", ["nan", "-0.1", "1.1"])
def test_evaluator_rejects_invalid_v2_gate_at_cli(monkeypatch, tmp_path, value):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(tmp_path / "model.skops"),
            "--width-override-confidence",
            "0.38",
            "--v2-gate-confidence",
            value,
        ],
    )
    with pytest.raises(SystemExit):
        eval_script.main()
