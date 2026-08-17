import numpy as np
import pytest

from scripts.benchmark_two_stage_mitdb import _fit_group_calibrated


def test_group_calibration_requires_multiple_records_and_two_classes():
    X = np.vstack([
        np.ones((6, 4)),
        -np.ones((6, 4)),
    ])
    y = np.array([1] * 6 + [0] * 6)
    groups = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3, dtype=object)
    model = _fit_group_calibrated(X, y, groups, target_recall=0.95, seed=42)
    assert model.metadata.calibration_method == "held_out_record_group"
    assert model.metadata.calibration_candidates > 0


def test_group_calibration_rejects_too_few_records():
    X = np.ones((6, 4))
    y = np.array([1, 1, 1, 0, 0, 0])
    groups = np.array(["a", "a", "b", "b", "b", "b"], dtype=object)
    with pytest.raises(ValueError, match="at least three training records"):
        _fit_group_calibrated(X, y, groups, target_recall=0.95, seed=42)
