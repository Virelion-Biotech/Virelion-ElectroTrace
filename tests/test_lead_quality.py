import numpy as np
import pytest

from electrotrace import lead_quality as lq


class _Model:
    fitted = True


def test_compute_lead_quality_uses_canonical_retained_stream(monkeypatch):
    signal = np.sin(np.linspace(0.0, 20.0, 1000))
    model = _Model()
    monkeypatch.setattr(
        lq,
        "select_signal_polarity",
        lambda *args, **kwargs: type(
            "D", (), {"polarity": "positive", "confidence": 0.8}
        )(),
    )
    monkeypatch.setattr(lq, "estimate_stage1_scale", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        lq,
        "_candidate_set",
        lambda *args, **kwargs: (
            np.array([100, 300, 500], dtype=int),
            np.array([2.0, 3.0, 4.0]),
        ),
    )
    monkeypatch.setattr(
        lq,
        "_candidate_features",
        lambda *args, **kwargs: (
            np.array([
                [1.0, 0.2],
                [1.0, 0.5],
                [1.0, 0.8],
            ]),
            ["amplitude_z", "qrs_band_fraction"],
        ),
    )
    monkeypatch.setattr(
        lq,
        "detect_r_peaks_two_stage",
        lambda *args, **kwargs: (
            np.array([100, 500], dtype=int),
            np.array([0.90, 0.98]),
        ),
    )
    retained, probs, quality = lq.compute_lead_quality(
        signal,
        100.0,
        model,
        scale_method="windowed_std",
        v2_gate_confidence=0.0,
        width_override_confidence=0.0,
        threshold=0.275,
    )
    assert retained.tolist() == [100, 500]
    assert probs.tolist() == pytest.approx([0.90, 0.98])
    assert quality.retained_probability_p50 == pytest.approx(0.94)
    assert quality.retained_qrs_band_fraction == pytest.approx(0.5)
    assert quality.retention_fraction == pytest.approx(2 / 3)
    assert quality.stage1_candidate_count == 3
    assert quality.retained_count == 2


def test_compute_lead_quality_fails_on_canonical_candidate_drift(monkeypatch):
    signal = np.ones(1000)
    model = _Model()
    monkeypatch.setattr(
        lq,
        "select_signal_polarity",
        lambda *args, **kwargs: type(
            "D", (), {"polarity": "positive", "confidence": 1.0}
        )(),
    )
    monkeypatch.setattr(lq, "estimate_stage1_scale", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        lq,
        "_candidate_set",
        lambda *args, **kwargs: (np.array([100]), np.array([2.0])),
    )
    monkeypatch.setattr(
        lq,
        "_candidate_features",
        lambda *args, **kwargs: (
            np.array([[1.0, 0.5]]),
            ["amplitude_z", "qrs_band_fraction"],
        ),
    )
    monkeypatch.setattr(
        lq,
        "detect_r_peaks_two_stage",
        lambda *args, **kwargs: (np.array([200]), np.array([0.9])),
    )
    with pytest.raises(RuntimeError, match="absent from reconstructed Stage-1"):
        lq.compute_lead_quality(
            signal,
            100.0,
            model,
            scale_method="windowed_std",
            v2_gate_confidence=0.0,
            width_override_confidence=0.0,
        )


def test_compute_lead_quality_empty_stream_is_zero_quality(monkeypatch):
    signal = np.ones(1000)
    model = _Model()
    monkeypatch.setattr(
        lq,
        "select_signal_polarity",
        lambda *args, **kwargs: type(
            "D", (), {"polarity": "positive", "confidence": 1.0}
        )(),
    )
    monkeypatch.setattr(lq, "estimate_stage1_scale", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        lq,
        "_candidate_set",
        lambda *args, **kwargs: (np.array([], dtype=int), np.array([])),
    )
    monkeypatch.setattr(
        lq,
        "detect_r_peaks_two_stage",
        lambda *args, **kwargs: (np.array([], dtype=int), np.array([])),
    )
    _, _, quality = lq.compute_lead_quality(
        signal,
        100.0,
        model,
        scale_method="windowed_std",
        v2_gate_confidence=0.0,
        width_override_confidence=0.0,
    )
    assert quality.retained_probability_p50 == 0.0
    assert quality.retained_qrs_band_fraction == 0.0
    assert quality.retention_fraction == 0.0
