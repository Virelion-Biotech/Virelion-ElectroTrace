import numpy as np
import pytest

from scripts import analyze_ltafdb_label_free_features as audit


class _Metadata:
    threshold = 0.5


class _Model:
    metadata = _Metadata()

    def predict_proba(self, features):
        return np.full(features.shape[0], 0.9, dtype=float)


def test_quantiles_and_relation_helpers():
    q = audit._quantiles(np.array([1.0, 2.0, 3.0]))
    assert q["p50"] == pytest.approx(2.0)
    assert audit._quantiles(np.array([])) is None
    assert audit._f1_relation(0.9, 0.8) == "channel0_better"
    assert audit._f1_relation(0.8, 0.9) == "channel1_better"
    assert audit._f1_relation(0.8, 0.8) == "tie"


def test_channel_quality_is_label_free_and_uses_canonical_detector(monkeypatch):
    signal = np.sin(np.linspace(0, 20, 1000))
    monkeypatch.setattr(
        audit,
        "select_signal_polarity",
        lambda *args, **kwargs: type("D", (), {"polarity": "positive", "confidence": 0.8})(),
    )
    monkeypatch.setattr(audit, "estimate_stage1_scale", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        audit,
        "_candidate_set",
        lambda *args, **kwargs: (
            np.array([100, 300, 500], dtype=int),
            np.array([2.0, 3.0, 4.0]),
        ),
    )
    monkeypatch.setattr(
        audit,
        "_candidate_features",
        lambda *args, **kwargs: (
            np.array([
                [1.0, 2.0, 0.04, 0.6, 1.0],
                [1.1, 2.1, 0.04, 0.7, 1.0],
                [1.2, 2.2, 0.05, 0.8, 1.0],
            ]),
            ["amplitude_z", "prominence_z", "width_s", "qrs_band_fraction", "local_rms"],
        ),
    )
    monkeypatch.setattr(
        audit,
        "detect_r_peaks_two_stage",
        lambda *args, **kwargs: (
            np.array([100, 500], dtype=int),
            np.array([0.95, 0.97]),
        ),
    )
    retained, quality = audit.channel_quality(
        signal,
        100.0,
        _Model(),
        v2_gate=0.0,
        width_gate=0.0,
        scale_method="windowed_std",
    )
    assert retained.tolist() == [100, 500]
    assert quality["retained_probability"]["p50"] == pytest.approx(0.96)
    assert quality["feature_medians_retained"]["qrs_band_fraction"] == pytest.approx(0.7)
    assert quality["retained_count"] == 2
    assert quality["retention_fraction"] == pytest.approx(2 / 3)


def test_channel_quality_rejects_nonfinite_signal():
    with pytest.raises(ValueError, match="finite one-dimensional"):
        audit.channel_quality(
            np.array([0.0, np.nan]),
            128.0,
            _Model(),
            v2_gate=0.0,
            width_gate=0.0,
            scale_method="windowed_std",
        )
