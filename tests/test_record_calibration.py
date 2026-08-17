import numpy as np
import pytest

from scripts.benchmark_two_stage_mitdb import _fit_group_calibrated


def test_group_calibration_requires_multiple_records_and_two_classes():
    X = np.vstack([
        np.ones((8, 4)),
        -np.ones((8, 4)),
    ])
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0] * 2)
    groups = np.repeat(["a", "b", "c", "d"], 4).astype(object)
    model, calibration_records = _fit_group_calibrated(X, y, groups, target_recall=0.95, seed=42)
    assert model.metadata.calibration_method == "held_out_record_group_stratified"
    assert model.metadata.calibration_candidates > 0
    assert calibration_records
    assert len(set(calibration_records)) == len(calibration_records)


def test_group_calibration_rejects_too_few_records():
    X = np.ones((6, 4))
    y = np.array([1, 1, 1, 0, 0, 0])
    groups = np.array(["a", "a", "b", "b", "b", "b"], dtype=object)
    with pytest.raises(ValueError, match="at least three training records"):
        _fit_group_calibrated(X, y, groups, target_recall=0.95, seed=42)
