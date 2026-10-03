import numpy as np
import pytest

import electrotrace.validation_detectors as vd
from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features
from electrotrace.validation_detectors import (
    DEFAULT_V2_GATE_CONFIDENCE,
    DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
    detect_r_peaks_two_stage,
    select_signal_polarity,
)


def _signal(n: int = 2000) -> np.ndarray:
    return np.random.default_rng(0).normal(0.0, 0.05, n)


def _fake_candidate_set(pos_count: int, neg_count: int):
    calls = {"n": 0}

    def fake(z, fs_hz, scale, **kwargs):
        calls["n"] += 1
        n = pos_count if calls["n"] == 1 else neg_count
        peaks = np.arange(n, dtype=int) * 10 + 100
        return peaks, np.full(n, scale, dtype=float)

    return fake, calls


def test_default_v2_gate_preserves_historical_behavior():
    assert DEFAULT_V2_GATE_CONFIDENCE == 0.15


def test_v2_gate_controls_fallback(monkeypatch):
    fake_candidates, calls = _fake_candidate_set(9, 10)  # raw confidence 0.10
    monkeypatch.setattr(vd, "_candidate_set", fake_candidates)
    monkeypatch.setattr(vd, "estimate_stage1_scale", lambda z, fs, method: 1.0)

    import electrotrace.polarity_v2 as pv2

    v2_calls = {"n": 0}

    def fake_v2(signal, fs_hz):
        v2_calls["n"] += 1
        return pv2.PolarityV2Decision(
            polarity="negative",
            confidence=0.99,
            positive_score=0.0,
            negative_score=0.0,
            qrs_events=10,
            positive_events=0,
            negative_events=10,
            ambiguous_events=0,
            rr_regularity=1.0,
            qrs_band_energy_ratio=1.0,
        )

    monkeypatch.setattr(pv2, "select_signal_polarity_v2", fake_v2)

    below = select_signal_polarity(
        _signal(),
        257.0,
        v2_gate_confidence=0.05,
        width_override_confidence=0.0,
    )
    assert below.confidence == pytest.approx(0.10)
    assert v2_calls["n"] == 0

    calls["n"] = 0
    historical = select_signal_polarity(
        _signal(),
        257.0,
        v2_gate_confidence=DEFAULT_V2_GATE_CONFIDENCE,
        width_override_confidence=0.0,
    )
    assert historical.polarity == "negative"
    assert historical.confidence == pytest.approx(0.99)
    assert v2_calls["n"] == 1


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("v2_nan", {"v2_gate_confidence": float("nan")}),
        ("v2_negative", {"v2_gate_confidence": -0.01}),
        ("v2_high", {"v2_gate_confidence": 1.01}),
        ("width_nan", {"width_override_confidence": float("nan")}),
        ("width_negative", {"width_override_confidence": -0.01}),
        ("width_high", {"width_override_confidence": 1.01}),
    ],
)
def test_confidence_gates_fail_closed_outside_unit_interval(name, kwargs):
    with pytest.raises(ValueError, match="between 0 and 1"):
        select_signal_polarity(_signal(), 257.0, **kwargs)


def _tiny_two_stage_model():
    fs = 360.0
    t = np.arange(7200) / fs
    signal = 0.05 * np.sin(2 * np.pi * 1.2 * t)
    candidates = np.array([900, 1800, 2700, 3600, 4500, 5400, 6300])
    for peak in candidates:
        signal[peak - 4 : peak + 5] += np.hanning(9) * 2.0
    features, names = _candidate_features(signal, fs, candidates, np.ones(len(candidates)))
    model = CandidateSuppressor().fit(
        np.vstack([features, features + 0.01]),
        np.array([1] * len(candidates) + [0] * len(candidates)),
        target_recall=0.9,
        n_estimators=20,
    )
    model.feature_names = names
    return model, signal, fs


def test_two_stage_threads_v2_gate(monkeypatch):
    model, signal, fs = _tiny_two_stage_model()
    seen = []
    real = vd.select_signal_polarity

    def spy(*args, **kwargs):
        seen.append(kwargs.get("v2_gate_confidence"))
        return real(*args, **kwargs)

    monkeypatch.setattr(vd, "select_signal_polarity", spy)
    detect_r_peaks_two_stage(
        signal,
        fs,
        model,
        polarity="adaptive",
        v2_gate_confidence=0.07,
    )
    assert 0.07 in seen


def test_two_stage_default_is_explicit_defaults():
    model, signal, fs = _tiny_two_stage_model()
    default_peaks, default_prob = detect_r_peaks_two_stage(
        signal,
        fs,
        model,
        polarity="adaptive",
    )
    explicit_peaks, explicit_prob = detect_r_peaks_two_stage(
        signal,
        fs,
        model,
        polarity="adaptive",
        width_override_confidence=DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
        v2_gate_confidence=DEFAULT_V2_GATE_CONFIDENCE,
    )
    assert np.array_equal(default_peaks, explicit_peaks)
    assert np.allclose(default_prob, explicit_prob)
