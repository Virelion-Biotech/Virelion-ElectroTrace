import numpy as np

from electrotrace.qrs_delineation import delineate_qrs


def synthetic_qrs(fs: float = 250.0) -> tuple[np.ndarray, int]:
    t = np.arange(0.0, 2.0, 1.0 / fs)
    x = 0.02 * np.sin(2 * np.pi * 1.0 * t)
    center = int(round(1.0 * fs))
    # Finite-width QRS-like pulse with short edges.
    shape = np.array([0.15, 0.35, 0.75, 1.25, 1.7, 1.25, 0.75, 0.35, 0.15])
    x[center - 4:center + 5] += shape
    return x, center


def test_qrs_delineator_is_deterministic_and_contains_center():
    x, center = synthetic_qrs()
    first = delineate_qrs(x, 250.0, center)
    second = delineate_qrs(x, 250.0, center)
    assert first == second
    assert first.onset <= center <= first.offset
    assert first.offset - first.onset > 0


def test_qrs_delineator_inverted_signal_keeps_boundaries_valid():
    x, center = synthetic_qrs()
    result = delineate_qrs(-x, 250.0, center)
    assert result.onset <= center <= result.offset
    assert result.offset - result.onset > 0


def test_qrs_delineator_rejects_invalid_center():
    x, _ = synthetic_qrs()
    for center in (-1, len(x)):
        try:
            delineate_qrs(x, 250.0, center)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
