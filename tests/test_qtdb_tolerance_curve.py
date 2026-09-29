from types import SimpleNamespace

import numpy as np
import pytest

from scripts import validate_qtdb as qtdb


class _WFDB:
    def __init__(self, annotations):
        self.annotations = annotations

    def rdann(self, path, extension):
        return self.annotations[extension]


def _ann(samples, symbols, nums):
    return SimpleNamespace(
        sample=np.asarray(samples, dtype=int),
        symbol=list(symbols),
        num=np.asarray(nums, dtype=int),
    )


def test_locked_protocol_uses_prespecified_tolerance_grid():
    protocol = qtdb.load_locked_protocol()
    assert protocol["primary_endpoint"]["tolerances_ms"] == [20, 40, 60, 80, 100]
    assert protocol["uncertainty"]["bootstrap_unit"] == "record"
    assert protocol["dataset"]["expected_records"] == 105
    assert protocol["dataset"]["expected_interobserver_records"] == 11


def test_qrs_reference_pairs_only_num1_qrs_boundaries(tmp_path):
    wfdb = _WFDB(
        {
            "q1c": _ann(
                [10, 20, 30, 40, 50, 60, 70, 80],
                ["(", ")", "(", ")", "(", ")", "(", ")"],
                [0, 0, 1, 1, 2, 2, 1, 1],
            )
        }
    )
    onset, offset = qtdb.qrs_reference(wfdb, tmp_path / "record", "q1c")
    assert onset.tolist() == [30, 70]
    assert offset.tolist() == [40, 80]


@pytest.mark.parametrize(
    "annotation,match",
    [
        (_ann([10], ["("], [1]), "unmatched final"),
        (_ann([10], [")"], [1]), "offset without onset"),
        (_ann([10, 11, 20], ["(", "(", ")"], [1, 1, 1]), "duplicate/unmatched"),
        (_ann([20, 10], ["(", ")"], [1, 1]), "non-positive"),
    ],
)
def test_qrs_reference_fails_closed_on_malformed_pairs(tmp_path, annotation, match):
    wfdb = _WFDB({"q1c": annotation})
    with pytest.raises(ValueError, match=match):
        qtdb.qrs_reference(wfdb, tmp_path / "record", "q1c")


def test_tolerance_curve_requires_boundary_found():
    counts = qtdb.tolerance_counts(
        np.asarray([5.0, 5.0, 30.0]),
        np.asarray([5.0, 30.0, 5.0]),
        np.asarray([True, False, True]),
        np.asarray([True, True, False]),
        [20, 40],
    )
    assert counts["20"]["eligible"] == 3
    assert counts["20"]["onset_success"] == 1
    assert counts["20"]["offset_success"] == 1
    assert counts["20"]["both_success"] == 1
    assert counts["40"]["onset_success"] == 2
    assert counts["40"]["offset_success"] == 2
    assert counts["40"]["both_success"] == 1


def test_match_indices_is_one_to_one():
    detected = np.asarray([100, 200, 205, 400], dtype=int)
    reference = np.asarray([102, 204, 399], dtype=int)
    pairs = qtdb.match_indices(detected, reference, 1000.0, 5.0)
    assert pairs == [(0, 0), (1, 1), (3, 2)]


def test_record_bootstrap_uses_record_counts_not_beat_resampling():
    records = [
        {
            "primary": {
                "tolerance_counts": {
                    "20": {
                        "eligible": 10,
                        "onset_success": 10,
                        "offset_success": 10,
                        "both_success": 10,
                    }
                }
            }
        },
        {
            "primary": {
                "tolerance_counts": {
                    "20": {
                        "eligible": 90,
                        "onset_success": 0,
                        "offset_success": 0,
                        "both_success": 0,
                    }
                }
            }
        },
    ]
    result = qtdb.aggregate_primary_curve(
        records,
        [20],
        replicates=1000,
        seed=42,
        interval=(2.5, 97.5),
    )
    both = result["20"]["both"]
    assert both["numerator"] == 10
    assert both["denominator"] == 100
    assert both["estimate"] == pytest.approx(0.10)
    lo, hi = both["record_bootstrap_interval"]
    assert 0.0 <= lo <= both["estimate"] <= hi <= 1.0


def test_validate_detected_rejects_duplicates_and_fractional_samples():
    with pytest.raises(ValueError, match="strictly increasing"):
        qtdb.validate_detected([10, 10], 100, "r")
    with pytest.raises(ValueError, match="finite integers"):
        qtdb.validate_detected([10.5], 100, "r")
    with pytest.raises(ValueError, match="outside signal support"):
        qtdb.validate_detected([100], 100, "r")
