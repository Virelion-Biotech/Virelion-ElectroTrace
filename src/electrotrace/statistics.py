"""Transparent statistics helpers with optional experimental-unit aggregation."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import stats


def _finite(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def _unit_reduce(values, unit_ids, label: str) -> tuple[np.ndarray, int]:
    if unit_ids is None:
        x = _finite(values)
        return x, int(len(x))
    raw_values = np.asarray(values, dtype=float)
    ids = np.asarray(unit_ids)
    if raw_values.ndim != 1 or ids.ndim != 1 or len(raw_values) != len(ids):
        raise ValueError(f"{label} values and unit_ids must be one-dimensional and equal length")
    if len(raw_values) == 0:
        return np.asarray([], dtype=float), 0
    buckets: dict[object, list[float]] = defaultdict(list)
    for value, unit in zip(raw_values, ids):
        if not np.isfinite(value):
            continue
        try:
            hash(unit)
        except TypeError as exc:
            raise ValueError(f"{label} unit_ids must contain hashable identifiers") from exc
        buckets[unit].append(float(value))
    reduced = np.asarray([np.mean(v) for v in buckets.values()], dtype=float)
    return reduced, int(len(buckets))


def compare_groups(
    values_a: list[float],
    values_b: list[float],
    unit_ids_a=None,
    unit_ids_b=None,
) -> dict:
    """Compare two groups, optionally aggregating repeated observations by experimental unit.

    When unit IDs are supplied, all finite observations from each unit are averaged before
    inference. This prevents treating multiple beats/measurements from one subject as
    independent observations by accident.
    """
    a, units_a = _unit_reduce(values_a, unit_ids_a, "group_a")
    b, units_b = _unit_reduce(values_b, unit_ids_b, "group_b")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least two finite experimental units/observations")
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    cohens_d = float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else None
    unit_mode = unit_ids_a is not None or unit_ids_b is not None
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "n_units_a": units_a if unit_ids_a is not None else int(len(a)),
        "n_units_b": units_b if unit_ids_b is not None else int(len(b)),
        "n_observations_a": int(np.asarray(values_a).size),
        "n_observations_b": int(np.asarray(values_b).size),
        "unit_of_analysis": "experimental_unit_mean" if unit_mode else "observation",
        "pseudoreplication_warning": not unit_mode,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "welch_t": float(t.statistic),
        "welch_p": float(t.pvalue),
        "mann_whitney_u": float(u.statistic),
        "mann_whitney_p": float(u.pvalue),
        "cohens_d": cohens_d,
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if len(p) == 0:
        return []
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be finite numbers between 0 and 1")
    n = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()
