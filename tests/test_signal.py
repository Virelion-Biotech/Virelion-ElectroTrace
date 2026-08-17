import numpy as np
import pytest

from electrotrace.signal import FilterConfigurationError, apply_pipeline


def test_filter_does_not_mutate_raw_data():
    fs = 500.0
    x = np.linspace(0, 2, int(fs * 2), endpoint=False)
    raw = np.sin(2 * np.pi * 1 * x) + 0.1 * np.sin(2 * np.pi * 80 * x)
    original = raw.copy()
    filtered = apply_pipeline(raw, fs, lowpass_hz=40)
    assert np.array_equal(raw, original)
    assert filtered.shape == raw.shape
    assert not np.array_equal(filtered, raw)


def test_invalid_cutoff_rejected():
    raw = np.ones(1000)
    with pytest.raises(FilterConfigurationError):
        apply_pipeline(raw, 500, lowpass_hz=300)


def test_band_order_rejected():
    raw = np.ones(1000)
    with pytest.raises(FilterConfigurationError):
        apply_pipeline(raw, 500, highpass_hz=40, lowpass_hz=10)
