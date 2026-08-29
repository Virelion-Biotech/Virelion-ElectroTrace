import numpy as np

from scripts.benchmark_two_stage_mitdb import _fit_group_calibrated


def test_group_calibration_excludes_calibration_records_from_model_fit():
    X = np.vstack([np.ones((8, 4)), -np.ones((8, 4))])
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0] * 2)
    groups = np.repeat(["a", "b", "c", "d"], 4).astype(object)

    model, fit_records, calibration_records = _fit_group_calibrated(
        X, y, groups, target_recall=0.95, seed=42
    )

    assert set(fit_records).isdisjoint(calibration_records)
    assert set(fit_records) | set(calibration_records) == set(groups)
    assert model.metadata.calibration_method == "held_out_record_group_stratified_f1"
    assert model.metadata.n_training_candidates == int(np.sum(np.isin(groups, fit_records)))
