import numpy as np
import pytest

from scripts.benchmark_two_stage_mitdb import _fit_group_calibrated


def test_group_calibration_requires_multiple_records_and_two_classes():
    X = np.vstack([np.ones((12, 3)), -np.ones((12, 3))])
    y = np.array([1, 1, 0, 0, 1, 0] * 4)
    groups = np.repeat(["r1", "r2", "r3", "r4"], 6).astype(object)

    model, fit_records, calibration_records = _fit_group_calibrated(
        X, y, groups, target_recall=0.9, seed=7
    )
    assert len(fit_records) >= 2
    assert len(calibration_records) >= 1
    assert set(fit_records).isdisjoint(calibration_records)
    assert model.metadata.calibration_method == "held_out_record_group_stratified_f1"


def test_group_calibration_rejects_too_few_records():
    X = np.ones((6, 2))
    y = np.array([1, 0, 1, 0, 1, 0])
    groups = np.array(["a", "a", "b", "b", "a", "b"], dtype=object)
    with pytest.raises(ValueError):
        _fit_group_calibrated(X, y, groups, target_recall=0.9, seed=0)
