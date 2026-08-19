import numpy as np

from electrotrace.polarity_v2 import PolarityV2Decision, select_signal_polarity_v2


def test_polarity_v2_is_deterministic_for_positive_signal():
    fs = 360.0
    t = np.arange(0, 10, 1 / fs)
    signal = 0.02 * np.sin(2 * np.pi * 1.1 * t)
    for center in np.arange(0.5, 9.5, 0.9):
        i = int(round(center * fs))
        signal[i - 2:i + 3] += np.array([0.8, 1.5, 2.2, 1.4, 0.7])
    first = select_signal_polarity_v2(signal, fs)
    second = select_signal_polarity_v2(signal, fs)
    assert first == second
    assert isinstance(first, PolarityV2Decision)
    assert first.qrs_events > 0


def test_polarity_v2_handles_inverted_signal_consistently():
    fs = 360.0
    t = np.arange(0, 10, 1 / fs)
    signal = 0.01 * np.sin(2 * np.pi * 1.0 * t)
    for center in np.arange(0.5, 9.5, 0.9):
        i = int(round(center * fs))
        signal[i - 2:i + 3] -= np.array([0.8, 1.5, 2.2, 1.4, 0.7])
    decision = select_signal_polarity_v2(signal, fs)
    assert decision.qrs_events > 0
    assert decision.polarity in {"positive", "negative"}
    assert decision.negative_events >= decision.positive_events or decision.negative_score > decision.positive_score


def test_polarity_v2_rejects_nonfinite_signal():
    signal = np.ones(100)
    signal[10] = np.nan
    try:
        select_signal_polarity_v2(signal, 360.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
