"""Regression tests for two concrete two-stage detector wiring bugs.

Bug 1: CandidateSuppressor.filter_candidates returned threshold-filtered
candidates but the full probability vector. CLI CSV construction zips the
arrays, so probabilities became shifted/misaligned whenever threshold > 0.

Bug 2: the detector registry advertised a callable returning np.ndarray but
the electrotrace-two-stage lambda returned the raw (peaks, probabilities)
tuple. Generic validate/bench paths consequently failed.
"""
import numpy as np
import pytest

from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features
from electrotrace.detectors import discover_detectors
from electrotrace.validation import validate_record
from electrotrace.validation_detectors import detect_r_peaks_two_stage

FS = 360.0


def _signal_with_peaks_and_t_waves(peaks, fs=FS, n=None):
    n = n or int(peaks[-1] + 2000)
    x = 0.02 * np.sin(2 * np.pi * 1.2 * np.arange(n) / fs)
    for p in peaks:
        x[p - 4 : p + 5] += np.hanning(9) * 2.0
        q = p + 150
        if q + 7 < n:
            x[q - 6 : q + 7] += np.hanning(13) * 0.7
    return x


def _model_with_midrange_threshold(peaks, signal):
    X, names = _candidate_features(signal, FS, peaks, np.ones(len(peaks)))
    X_train = np.vstack([X, X * 0 + 0.001])
    y_train = np.array([1] * len(peaks) + [0] * len(peaks))
    model = CandidateSuppressor().fit(X_train, y_train, target_recall=0.9, n_estimators=40)
    model.feature_names = names
    return model


def test_cli_two_stage_csv_probability_matches_each_retained_peaks_own_score():
    from electrotrace.cli import _two_stage

    peaks = np.array([900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 8100])
    signal = _signal_with_peaks_and_t_waves(peaks)
    model = _model_with_midrange_threshold(peaks, signal)
    model.metadata = model.metadata.__class__(**{**model.metadata.to_dict(), "threshold": 0.3})

    cli_peaks, cli_probabilities = _two_stage(signal, FS, model, "positive", "windowed_std")
    assert len(cli_peaks) == len(cli_probabilities)

    all_candidates, all_probabilities = detect_r_peaks_two_stage(
        signal, FS, model, polarity="positive", scale_method="windowed_std", threshold=0.0
    )
    truth = dict(zip(all_candidates.tolist(), all_probabilities.tolist()))

    for peak, reported_prob in zip(cli_peaks.tolist(), cli_probabilities.tolist()):
        assert reported_prob == pytest.approx(truth[peak])
        assert reported_prob >= 0.3


def test_electrotrace_two_stage_detector_spec_returns_plain_array_not_tuple():
    peaks = np.array([900, 1800, 2700, 3600, 4500, 5400, 6300])
    signal = _signal_with_peaks_and_t_waves(peaks)
    model = _model_with_midrange_threshold(peaks, signal)

    spec = discover_detectors(model=model, polarity="positive", scale_method="windowed_std")[
        "electrotrace-two-stage"
    ]
    result = spec.detector(signal, FS)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    expected_peaks, _ = detect_r_peaks_two_stage(
        signal, FS, model, polarity="positive", scale_method="windowed_std"
    )
    assert np.array_equal(result, expected_peaks)


def test_validate_record_does_not_crash_with_electrotrace_two_stage_spec(monkeypatch):
    import electrotrace.validation as validation_module

    peaks = np.array([900, 1800, 2700, 3600, 4500, 5400, 6300])
    signal = _signal_with_peaks_and_t_waves(peaks)
    model = _model_with_midrange_threshold(peaks, signal)
    spec = discover_detectors(model=model, polarity="positive", scale_method="windowed_std")[
        "electrotrace-two-stage"
    ]

    monkeypatch.setattr(validation_module, "_record_signal", lambda *a, **k: (signal, FS))
    monkeypatch.setattr(validation_module, "load_reference_annotations", lambda *a, **k: peaks)

    result = validate_record("dummy_record", spec.detector)
    assert result.metrics.reference_count == len(peaks)
    assert result.metrics.sensitivity > 0.0
