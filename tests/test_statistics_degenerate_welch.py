"""Undefined Welch tests remain explicit and standards-compliant JSON."""
import json
import warnings

import pytest
from scipy import stats

from electrotrace.statistics import compare_groups


@pytest.mark.parametrize("a,b", [
    ([1., 1.], [1., 1.]),
    ([1., 1.], [2., 2.]),
])
def test_constant_groups_have_no_defined_welch_test(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = compare_groups(a, b)
    assert result["welch_t"] is None
    assert result["welch_p"] is None
    assert result["welch_status"] == "undefined_zero_variance"
    assert result["cohens_d"] is None
    assert result["mann_whitney_p"] is not None
    json.dumps(result, allow_nan=False)


def test_defined_welch_test_preserves_scipy_values():
    a, b = [1., 2., 4.], [2., 4., 7.]
    expected = stats.ttest_ind(a, b, equal_var=False)
    result = compare_groups(a, b)
    assert result["welch_t"] == pytest.approx(expected.statistic)
    assert result["welch_p"] == pytest.approx(expected.pvalue)
    assert result["welch_status"] == "computed"
    json.dumps(result, allow_nan=False)


def test_zero_variance_checked_after_experimental_unit_aggregation():
    result = compare_groups([0., 2., 0., 2.], [0., 2., 0., 2.],
                            unit_ids_a=["a", "a", "b", "b"],
                            unit_ids_b=["c", "c", "d", "d"])
    assert result["welch_status"] == "undefined_zero_variance"
    assert result["n_units_a"] == result["n_units_b"] == 2
    json.dumps(result, allow_nan=False)


def test_all_tied_mann_whitney_has_exact_degenerate_null():
    result = compare_groups([1., 1., 1.], [1., 1.])
    assert result["mann_whitney_u"] == 3.
    assert result["mann_whitney_p"] == 1.
    assert result["mann_whitney_status"] == "all_values_identical"
    json.dumps(result, allow_nan=False)
