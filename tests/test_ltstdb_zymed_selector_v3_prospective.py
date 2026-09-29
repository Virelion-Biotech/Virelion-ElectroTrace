from types import SimpleNamespace

import numpy as np
import pytest

from electrotrace.lead_quality import LeadQuality
from scripts import evaluate_ltstdb_zymed_selector_v3_prospective as lt


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


def _quality(count):
    return LeadQuality(
        retained_probability_p50=0.9,
        retained_qrs_band_fraction=0.5,
        retention_fraction=0.5,
        stage1_candidate_count=max(count, 1),
        retained_count=count,
        selected_polarity="positive",
        polarity_confidence=1.0,
    )


def test_protocol_is_locked_and_exact_zymed_subset():
    assert lt._sv.git_blob_sha(lt.PROTOCOL_PATH) == lt.EXPECTED_PROTOCOL_GIT_BLOB_SHA
    p = lt.load_locked_protocol()
    assert len(p["dataset"]["records"]) == 18
    assert p["dataset"]["records"][0] == "s30661"
    assert p["dataset"]["records"][-1] == "s30801"
    assert p["dataset"]["candidate_channels"] == [0, 1]
    assert p["dataset"]["prohibited_channels"] == [2]
    assert p["dataset"]["window"]["stop_sample_exclusive"] == 450000


def test_verify_dataset_extracts_only_canonical_s3_subset(monkeypatch, tmp_path):
    p = lt.load_locked_protocol()
    all_records = ["s20011", *p["dataset"]["records"]]
    (tmp_path / "RECORDS").write_text("\n".join(all_records) + "\n", encoding="utf-8")
    for record in p["dataset"]["records"]:
        for suffix in lt.REQUIRED_SUFFIXES:
            (tmp_path / f"{record}{suffix}").write_bytes(b"x")
    monkeypatch.setattr(
        lt.wfdb, "rdheader",
        lambda path: SimpleNamespace(n_sig=3, fs=250.0, sig_len=999999),
    )
    hashes = lt.verify_dataset(tmp_path, p)
    assert len(hashes) == 1 + 18 * 3


def test_evaluate_record_never_loads_channel2_and_selects_before_annotations(monkeypatch, tmp_path):
    p = lt.load_locked_protocol()
    model = _Model(p)
    events = []
    signal = np.zeros((450000, 2), dtype=float)

    def fake_rdrecord(*args, **kwargs):
        events.append("signal")
        assert kwargs["channels"] == [0, 1]
        assert kwargs["sampfrom"] == 0
        assert kwargs["sampto"] == 450000
        return SimpleNamespace(fs=250.0, p_signal=signal, sig_name=["A", "B"])
    monkeypatch.setattr(lt.wfdb, "rdrecord", fake_rdrecord)

    calls = {"n": 0}
    def fake_quality(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            retained = np.arange(600, dtype=int)
        else:
            retained = np.arange(2400, dtype=int)
        return retained, np.full(len(retained), 0.9), _quality(len(retained))
    monkeypatch.setattr(lt, "compute_lead_quality", fake_quality)

    def fake_choose(r0, r1, **kwargs):
        events.append("choose")
        assert r0 == pytest.approx(20.0)
        assert r1 == pytest.approx(80.0)
        return 1
    monkeypatch.setattr(lt, "choose_two_lead_channel_v3", fake_choose)

    def fake_rdann(*args, **kwargs):
        events.append("annotation")
        assert kwargs == {"sampfrom": 0, "sampto": 449999}
        return SimpleNamespace(sample=np.array([100, 300, 500]), symbol=["N", "N", "N"])
    monkeypatch.setattr(lt.wfdb, "rdann", fake_rdann)

    result, payload, audit = lt.evaluate_record(
        tmp_path, "s30661", protocol=p, model=model, v2_gate=0.0, width_gate=0.0
    )
    assert events.index("choose") < events.index("annotation")
    assert payload["selected_channel"] == 1
    assert payload["third_channel_loaded"] is False
    assert audit["third_channel_loaded"] is False
    assert result.metrics.f1 >= 0.0


def test_selector_window_rate_is_30_minutes():
    p = lt.load_locked_protocol()
    assert p["dataset"]["window"]["duration_seconds"] == 1800
