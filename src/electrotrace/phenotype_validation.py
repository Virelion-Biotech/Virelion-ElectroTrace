"""Generic quality control for extracted ECG phenotypes.

The checks here deliberately avoid hard-coded clinical cutoffs. They validate
mathematical consistency, missingness, uniqueness, and experimental-unit
aggregation so downstream studies can add domain-specific thresholds without
silently changing the core toolkit.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

PHENOTYPE_SCHEMA_VERSION = "beat-phenotype-v1"


def quality_report(
    phenotypes: Sequence[Mapping[str, Any]],
    *,
    required_fields: Sequence[str] = ("r_index", "r_time_s", "heart_rate_bpm", "r_amplitude"),
    consistency_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Check structural and arithmetic validity without imposing clinical ranges."""
    rows = list(phenotypes)
    if consistency_tolerance < 0 or not np.isfinite(consistency_tolerance):
        raise ValueError("consistency_tolerance must be finite and non-negative")
    missing = {field: 0 for field in required_fields}
    nonfinite = {field: 0 for field in required_fields}
    invalid_r_indices = 0
    duplicate_r_indices = 0
    inconsistent_hr = 0
    invalid_rr = 0
    times: list[float] = []
    r_indices: list[int] = []
    for row in rows:
        for field in required_fields:
            value = row.get(field)
            if value is None:
                missing[field] += 1
            else:
                try:
                    if not np.isfinite(float(value)):
                        nonfinite[field] += 1
                except (TypeError, ValueError):
                    nonfinite[field] += 1
        if row.get("r_index") is not None:
            try:
                numeric_index = float(row["r_index"])
                integer_index = int(numeric_index)
                if not np.isfinite(numeric_index) or numeric_index != integer_index or integer_index < 0:
                    invalid_r_indices += 1
                else:
                    r_indices.append(integer_index)
            except (TypeError, ValueError, OverflowError):
                invalid_r_indices += 1
        if row.get("r_time_s") is not None:
            try:
                times.append(float(row["r_time_s"]))
            except (TypeError, ValueError):
                pass
        rr = row.get("rr_prev_s")
        hr = row.get("heart_rate_bpm")
        if rr is not None:
            try:
                if not np.isfinite(float(rr)) or float(rr) <= 0:
                    invalid_rr += 1
                elif hr is not None and np.isfinite(float(hr)):
                    expected = 60.0 / float(rr)
                    if abs(expected - float(hr)) > max(consistency_tolerance, abs(expected) * consistency_tolerance):
                        inconsistent_hr += 1
            except (TypeError, ValueError, ZeroDivisionError):
                invalid_rr += 1
    duplicate_r_indices = len(r_indices) - len(set(r_indices))
    time_nonmonotonic = int(any(b <= a for a, b in zip(times, times[1:])))
    missing_total = int(sum(missing.values()))
    nonfinite_total = int(sum(nonfinite.values()))
    return {
        "schema_version": PHENOTYPE_SCHEMA_VERSION,
        "n_rows": len(rows),
        "missing": missing,
        "missing_total": missing_total,
        "nonfinite": nonfinite,
        "nonfinite_total": nonfinite_total,
        "invalid_r_indices": invalid_r_indices,
        "duplicate_r_indices": duplicate_r_indices,
        "time_nonmonotonic": time_nonmonotonic,
        "invalid_rr": invalid_rr,
        "inconsistent_heart_rate": inconsistent_hr,
        "valid": not any((missing_total, nonfinite_total, invalid_r_indices, duplicate_r_indices, time_nonmonotonic, invalid_rr, inconsistent_hr)),
    }


def aggregate_by_unit(
    phenotypes: Sequence[Mapping[str, Any]],
    unit_ids: Sequence[str],
    *,
    value_fields: Sequence[str] = ("heart_rate_bpm", "rr_prev_s", "r_amplitude", "qrs_width_proxy_s"),
) -> list[dict[str, Any]]:
    """Aggregate beats within the declared experimental unit to avoid pseudoreplication."""
    rows = list(phenotypes)
    units = [str(x) for x in unit_ids]
    if len(rows) != len(units):
        raise ValueError("phenotypes and unit_ids must have equal length")
    if any(not unit.strip() for unit in units):
        raise ValueError("unit_ids must be non-empty")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row, unit in zip(rows, units):
        grouped[unit].append(row)
    output: list[dict[str, Any]] = []
    for unit in sorted(grouped):
        group = grouped[unit]
        item: dict[str, Any] = {"unit_id": unit, "n_beats": len(group)}
        for field in value_fields:
            values = []
            for row in group:
                value = row.get(field)
                if value is not None:
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        values.append(value)
            x = np.asarray(values, dtype=float)
            item[field] = {
                "n": int(x.size),
                "mean": float(np.mean(x)) if x.size else None,
                "median": float(np.median(x)) if x.size else None,
                "sd": float(np.std(x, ddof=1)) if x.size > 1 else None,
            }
        output.append(item)
    return output
