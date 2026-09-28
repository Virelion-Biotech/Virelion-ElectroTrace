import json
import sys
import types

import numpy as np
import pytest

from electrotrace.validation_detectors import select_signal_polarity
from scripts import derive_polarity_thresholds_mitdb_extended as script

FS = 257.0


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


def _peaks(n=20, step=300, start=200):
    return np.arange(start, start + n * step, step)


def _ambiguous_signal(peaks, offset=150):
    n = int(peaks[-1] + offset + 1000)
    signal = np.zeros(n)
    for peak in peaks:
        signal[peak - 4 : peak + 5] += np.hanning(9) * 2.0
        other = peak + offset
        signal[other - 4 : other + 5] -= np.hanning(9) * 2.0
    return signal


def _cache(record, raw_confidence, width_preferred=None, v2_polarity="negative", v2_confidence=0.9):
    return script.RecordCache(
        record=record,
        fs_hz=FS,
        reference=np.array([100, 200, 300]),
        pos=np.array([100, 200, 300]),
        neg=np.array([150, 250, 350]),
        pos_count=3,
        neg_count=3,
        ratio_polarity="positive",
        raw_confidence=raw_confidence,
        v2_polarity=v2_polarity,
        v2_confidence=v2_confidence,
        width_preferred=width_preferred,
    )


def test_decide_matches_real_selector_across_gate_grid(monkeypatch, tmp_path):
    peaks = _peaks()
    signal = _ambiguous_signal(peaks)
    _fake_wfdb(monkeypatch, {"999": (signal, FS, peaks)})
    cache = script.build_record_cache(tmp_path, "999", tolerance_ms=75.0)

    for v2_gate in (0.0, 0.05, 0.15, 0.3):
        for width_gate in (0.0, 0.1, 0.38):
            replica = script.decide(cache, v2_gate, width_gate)
            real = select_signal_polarity(
                signal,
                FS,
                v2_gate_confidence=v2_gate,
                width_override_confidence=width_gate,
            )
            assert replica[0] == real.polarity
            assert replica[1] == pytest.approx(real.confidence)


def test_non_identifying_dimension_is_not_mistaken_for_selected_value():
    v2_grid = [0.0, 0.1, 0.2]
    width_grid = [0.0, 0.3]
    results = [
        {
            "v2_gate_confidence": gate,
            "width_override_confidence": width,
            "mean_f1": 0.9 if width == 0.3 else 0.7,
        }
        for gate in v2_grid
        for width in width_grid
    ]
    assert script.find_non_identifying_dimensions(results, v2_grid, width_grid) == [
        "v2_gate_confidence"
    ]

    status = script._parameter_status(
        name="v2_gate_confidence",
        selected=0.0,
        historical=0.15,
        non_identifying=["v2_gate_confidence"],
        changed_records=["X"],
    )
    assert status["identified"] is False
    assert status["recommended_frozen_value"] == pytest.approx(0.15)


def test_engagement_distinguishes_reached_from_actual_width_override():
    same = _cache("SAME", 0.20, width_preferred="positive")
    changed = _cache("CHANGED", 0.20, width_preferred="negative")

    summary = script.engagement_summary(
        [same, changed],
        v2_gate=0.15,
        width_override=0.38,
    )
    assert summary["records_where_width_check_reached"] == ["CHANGED", "SAME"]
    assert summary["records_where_width_override_changes_final_polarity"] == ["CHANGED"]


def test_build_record_cache_refuses_locked_record(tmp_path):
    with pytest.raises(RuntimeError, match="locked held-out"):
        script.build_record_cache(tmp_path, "207", tolerance_ms=75.0)


def test_default_mode_requires_complete_standard_development_pool(monkeypatch, tmp_path):
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "derive_polarity_thresholds_mitdb_extended.py",
            "--mitdb-dir",
            str(tmp_path),
            "--output",
            str(out),
        ],
    )
    with pytest.raises(SystemExit, match="Incomplete MIT-BIH development pool"):
        script.main()


def test_custom_records_require_explicit_diagnostic_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "derive_polarity_thresholds_mitdb_extended.py",
            "--mitdb-dir",
            str(tmp_path),
            "--records",
            "X1",
            "X2",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(SystemExit, match="diagnostic-only"):
        script.main()


def test_diagnostic_subset_cannot_include_locked_record(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "derive_polarity_thresholds_mitdb_extended.py",
            "--mitdb-dir",
            str(tmp_path),
            "--records",
            "207",
            "X2",
            "--allow-diagnostic-subset",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(SystemExit, match="locked held-out"):
        script.main()


def test_invalid_grid_is_rejected_not_silently_filtered(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "derive_polarity_thresholds_mitdb_extended.py",
            "--mitdb-dir",
            str(tmp_path),
            "--records",
            "X1",
            "X2",
            "--allow-diagnostic-subset",
            "--v2-gate-grid",
            "nan",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(SystemExit, match="finite and between 0 and 1"):
        script.main()


def test_any_record_load_failure_aborts_derivation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        script,
        "build_record_cache",
        lambda root, name, tolerance_ms: (
            _cache(name, 0.9)
            if name == "X1"
            else (_ for _ in ()).throw(ValueError("broken annotation"))
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "derive_polarity_thresholds_mitdb_extended.py",
            "--mitdb-dir",
            str(tmp_path),
            "--records",
            "X1",
            "X2",
            "--allow-diagnostic-subset",
            "--v2-gate-grid",
            "0.0",
            "--width-override-grid",
            "0.0",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(SystemExit, match="Derivation aborted"):
        script.main()


def test_diagnostic_report_is_never_freeze_eligible(monkeypatch, tmp_path):
    caches = {
        "X1": _cache("X1", 0.8),
        "X2": _cache("X2", 0.9),
    }
    monkeypatch.setattr(
        script,
        "build_record_cache",
        lambda root, name, tolerance_ms: caches[name],
    )
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "derive_polarity_thresholds_mitdb_extended.py",
            "--mitdb-dir",
            str(tmp_path),
            "--records",
            "X1",
            "X2",
            "--allow-diagnostic-subset",
            "--v2-gate-grid",
            "0.0",
            "0.15",
            "--width-override-grid",
            "0.0",
            "0.38",
            "--output",
            str(out),
        ],
    )
    assert script.main() == 0
    report = json.loads(out.read_text())
    assert report["selection_status"] == "diagnostic_subset_only"
    assert report["freeze_eligible"] is False
    assert report["protocol"]["locked_heldout_labels_used"] is False
    assert report["protocol"]["incart_used"] is False
    assert report["evaluation_integrity"][
        "adaptive_polarity_mechanism_was_historically_informed_by_locked_data"
    ] is True
