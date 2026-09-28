#!/usr/bin/env python3
"""Derive adaptive-polarity confidence gates using MIT-BIH development data only.

This utility exists to address the threshold-selection part of Issue #14
without pretending that historical exposure of record 207 can be undone.
It never reads the 12 locked MIT-BIH records and has no INCART input.

Default mode is intentionally strict:
- exactly the 36 standard MIT-BIH records outside the locked 12-record split
  form the development pool;
- .hea, .dat, and .atr must exist for every development record;
- any load/annotation failure aborts derivation instead of silently shrinking
  the pool;
- the output distinguishes selected grid points from recommended frozen values
  when a parameter is non-identifying.

A custom --records pool is diagnostic-only and requires
--allow-diagnostic-subset. Such a report is never eligible to authorize a
non-default gate in the locked evaluator.

The optimization target is record-level Stage-1 F1. This is a development
procedure, not an independent validation study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import electrotrace.polarity_v2 as polarity_v2_module
import electrotrace.scale_estimation as scale_estimation_module
import electrotrace.validation_detectors as validation_detectors_module
from electrotrace import __version__
from electrotrace.polarity_v2 import select_signal_polarity_v2
from electrotrace.scale_estimation import estimate_stage1_scale
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, match_peaks
from electrotrace.validation_detectors import (
    DEFAULT_NEGATIVE_COUNT_RATIO,
    DEFAULT_V2_GATE_CONFIDENCE,
    DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
    DEFAULT_WIDTH_OVERRIDE_MIN_CANDIDATES,
    _candidate_set,
    _width_preferred_polarity,
)
from electrotrace.wfdb_records import load_annotated_record

MITDB_RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
)
LOCKED_HELDOUT_RECORDS = frozenset(
    {"105", "118", "122", "201", "207", "209", "214", "219", "230", "231", "232", "234"}
)
DEVELOPMENT_RECORDS = tuple(name for name in MITDB_RECORDS if name not in LOCKED_HELDOUT_RECORDS)
REQUIRED_SUFFIXES = (".hea", ".dat", ".atr")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hashes() -> dict[str, str]:
    paths = {
        "derive_polarity_thresholds_mitdb_extended.py": Path(__file__).resolve(),
        "electrotrace.polarity_v2": Path(polarity_v2_module.__file__).resolve(),
        "electrotrace.scale_estimation": Path(scale_estimation_module.__file__).resolve(),
        "electrotrace.validation_detectors": Path(validation_detectors_module.__file__).resolve(),
    }
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("numpy", "scipy", "wfdb"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "missing"
    return versions


def _validated_grid(values: list[float], name: str) -> list[float]:
    grid: list[float] = []
    for raw in values:
        value = float(raw)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} values must be finite and between 0 and 1")
        grid.append(value)
    grid = sorted(set(grid))
    if not grid:
        raise SystemExit(f"{name} must not be empty")
    return grid


def _assert_complete_standard_pool(mitdb_dir: Path) -> None:
    missing: list[str] = []
    for name in DEVELOPMENT_RECORDS:
        for suffix in REQUIRED_SUFFIXES:
            path = mitdb_dir / f"{name}{suffix}"
            if not path.is_file():
                missing.append(path.name)
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f" ... (+{len(missing) - 12} more)"
        raise SystemExit(
            "Incomplete MIT-BIH development pool. Fetch the full 48-record database before "
            f"derivation. Missing: {preview}{suffix}"
        )


@dataclass
class RecordCache:
    record: str
    fs_hz: float
    reference: np.ndarray
    pos: np.ndarray
    neg: np.ndarray
    pos_count: int
    neg_count: int
    ratio_polarity: str
    raw_confidence: float
    v2_polarity: str
    v2_confidence: float
    width_preferred: str | None


def build_record_cache(mitdb_dir: Path, name: str, tolerance_ms: float) -> RecordCache:
    if name in LOCKED_HELDOUT_RECORDS:
        raise RuntimeError(f"refusing to use locked held-out record {name}")

    annotated = load_annotated_record(
        mitdb_dir / name,
        beat_symbols=DEFAULT_BEAT_SYMBOLS,
        tolerance_ms=tolerance_ms,
        policy="error",
    )
    signal = annotated.signal
    z = signal - np.median(signal)
    scale = estimate_stage1_scale(z, annotated.fs_hz, method="windowed_std")
    if not np.isfinite(scale) or scale == 0:
        pos = neg = np.asarray([], dtype=int)
    else:
        pos, _ = _candidate_set(z, annotated.fs_hz, scale)
        neg, _ = _candidate_set(-z, annotated.fs_hz, scale)

    pos_count = len(pos)
    neg_count = len(neg)
    ratio = neg_count / max(pos_count, 1)
    ratio_polarity = (
        "negative"
        if pos_count > 0 and neg_count > 0 and ratio < DEFAULT_NEGATIVE_COUNT_RATIO
        else "positive"
    )
    raw_confidence = float(abs(pos_count - neg_count) / max(pos_count, neg_count, 1))
    v2 = select_signal_polarity_v2(signal, annotated.fs_hz)
    width_preferred = _width_preferred_polarity(
        z,
        pos,
        neg,
        min_candidates=DEFAULT_WIDTH_OVERRIDE_MIN_CANDIDATES,
    )
    return RecordCache(
        record=name,
        fs_hz=annotated.fs_hz,
        reference=annotated.reference,
        pos=pos,
        neg=neg,
        pos_count=pos_count,
        neg_count=neg_count,
        ratio_polarity=ratio_polarity,
        raw_confidence=raw_confidence,
        v2_polarity=v2.polarity,
        v2_confidence=float(v2.confidence),
        width_preferred=width_preferred,
    )


def decide(cache: RecordCache, v2_gate: float, width_override: float) -> tuple[str, float]:
    """Replicate select_signal_polarity's cheap branching from cached values."""
    polarity = cache.ratio_polarity
    confidence = cache.raw_confidence
    if confidence < v2_gate:
        polarity = cache.v2_polarity
        confidence = max(confidence, cache.v2_confidence)
    if confidence < width_override:
        if cache.width_preferred is not None and cache.width_preferred != polarity:
            polarity = cache.width_preferred
    return polarity, confidence


def score_grid_point(
    caches: list[RecordCache],
    v2_gate: float,
    width_override: float,
    tolerance_ms: float,
    *,
    include_records: bool = False,
) -> dict:
    rows = []
    for cache in caches:
        polarity, confidence = decide(cache, v2_gate, width_override)
        peaks = cache.neg if polarity == "negative" else cache.pos
        metrics = match_peaks(peaks, cache.reference, cache.fs_hz, tolerance_ms=tolerance_ms)
        rows.append(
            {
                "record": cache.record,
                "polarity": polarity,
                "confidence": confidence,
                "f1": metrics.f1,
                "sensitivity": metrics.sensitivity,
                "positive_predictive_value": metrics.positive_predictive_value,
                "candidate_count": int(len(peaks)),
            }
        )

    result = {
        "v2_gate_confidence": float(v2_gate),
        "width_override_confidence": float(width_override),
        "mean_f1": float(np.mean([row["f1"] for row in rows])),
        "mean_sensitivity": float(np.mean([row["sensitivity"] for row in rows])),
        "mean_ppv": float(np.mean([row["positive_predictive_value"] for row in rows])),
    }
    if include_records:
        result["records"] = rows
    return result


def find_non_identifying_dimensions(
    results: list[dict],
    v2_grid: list[float],
    width_grid: list[float],
) -> list[str]:
    """Return dimensions whose full tested range remains represented among global F1 optima."""
    best_f1 = max(item["mean_f1"] for item in results)
    tied = [item for item in results if abs(item["mean_f1"] - best_f1) < 1e-12]
    flagged: list[str] = []
    if {item["v2_gate_confidence"] for item in tied} >= set(v2_grid):
        flagged.append("v2_gate_confidence")
    if {item["width_override_confidence"] for item in tied} >= set(width_grid):
        flagged.append("width_override_confidence")
    return flagged


def engagement_summary(
    caches: list[RecordCache],
    v2_gate: float,
    width_override: float,
) -> dict[str, list[str]]:
    v2_reached: list[str] = []
    v2_changed_final: list[str] = []
    width_reached: list[str] = []
    width_changed_final: list[str] = []
    changed_vs_primary: list[str] = []
    changed_vs_historical: list[str] = []

    for cache in caches:
        primary = decide(cache, 0.0, 0.0)
        historical = decide(
            cache,
            DEFAULT_V2_GATE_CONFIDENCE,
            DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
        )
        selected = decide(cache, v2_gate, width_override)
        no_v2 = decide(cache, 0.0, width_override)
        no_width = decide(cache, v2_gate, 0.0)

        if cache.raw_confidence < v2_gate:
            v2_reached.append(cache.record)
        if selected[0] != no_v2[0]:
            v2_changed_final.append(cache.record)

        if no_width[1] < width_override and cache.width_preferred is not None:
            width_reached.append(cache.record)
        if selected[0] != no_width[0]:
            width_changed_final.append(cache.record)

        if selected[0] != primary[0]:
            changed_vs_primary.append(cache.record)
        if selected[0] != historical[0]:
            changed_vs_historical.append(cache.record)

    return {
        "records_where_v2_gate_reached": sorted(v2_reached),
        "records_where_v2_gate_changes_final_polarity": sorted(v2_changed_final),
        "records_where_width_check_reached": sorted(width_reached),
        "records_where_width_override_changes_final_polarity": sorted(width_changed_final),
        "records_changed_vs_primary_rule_alone": sorted(changed_vs_primary),
        "records_changed_vs_historical_defaults": sorted(changed_vs_historical),
    }


def _parameter_status(
    *,
    name: str,
    selected: float,
    historical: float,
    non_identifying: list[str],
    changed_records: list[str],
) -> dict:
    identified = name not in non_identifying and bool(changed_records)
    if identified:
        frozen = selected
        reason = "development F1 identifies this dimension and it changes final polarity on at least one record"
    elif name in non_identifying:
        frozen = historical
        reason = "global F1 optimum does not identify this dimension; historical default retained"
    else:
        frozen = historical
        reason = "selected operating point does not change final polarity; historical default retained"
    return {
        "identified": identified,
        "selected_grid_value": float(selected),
        "historical_default": float(historical),
        "recommended_frozen_value": float(frozen),
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mitdb-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument(
        "--v2-gate-grid",
        type=float,
        nargs="*",
        default=np.round(np.arange(0.0, 0.31, 0.01), 2).tolist(),
    )
    parser.add_argument(
        "--width-override-grid",
        type=float,
        nargs="*",
        default=np.round(np.arange(0.0, 0.61, 0.02), 2).tolist(),
    )
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Diagnostic-only custom pool; requires --allow-diagnostic-subset and cannot authorize a frozen override.",
    )
    parser.add_argument("--allow-diagnostic-subset", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.tolerance_ms) or args.tolerance_ms <= 0:
        raise SystemExit("--tolerance-ms must be positive and finite")

    v2_grid = _validated_grid(args.v2_gate_grid, "--v2-gate-grid")
    width_grid = _validated_grid(args.width_override_grid, "--width-override-grid")

    diagnostic_subset = args.records is not None
    if diagnostic_subset and not args.allow_diagnostic_subset:
        raise SystemExit("--records is diagnostic-only; also pass --allow-diagnostic-subset")
    if diagnostic_subset:
        requested = list(dict.fromkeys(str(item) for item in args.records))
        forbidden = sorted(set(requested) & LOCKED_HELDOUT_RECORDS)
        if forbidden:
            raise SystemExit(
                "refusing diagnostic subset containing locked held-out records: "
                + ", ".join(forbidden)
            )
        pool_names = requested
        selection_status = "diagnostic_subset_only"
        freeze_eligible = False
    else:
        _assert_complete_standard_pool(args.mitdb_dir)
        pool_names = list(DEVELOPMENT_RECORDS)
        selection_status = "development_only_full_pool"
        freeze_eligible = True

    if len(pool_names) < 2:
        raise SystemExit("At least two development records are required.")

    caches: list[RecordCache] = []
    input_hashes: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    for index, name in enumerate(pool_names, start=1):
        try:
            cache = build_record_cache(args.mitdb_dir, name, args.tolerance_ms)
        except Exception as exc:
            failures.append({"record": name, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(pool_names)}] {name}: FAILED: {exc}", flush=True)
            continue

        caches.append(cache)
        for suffix in REQUIRED_SUFFIXES:
            source = args.mitdb_dir / f"{name}{suffix}"
            if source.is_file():
                input_hashes[f"{name}{suffix}"] = sha256_file(source)
        print(
            f"[{index}/{len(pool_names)}] {name}: raw_confidence={cache.raw_confidence:.3f} "
            f"pos={cache.pos_count} neg={cache.neg_count} "
            f"v2={cache.v2_polarity}/{cache.v2_confidence:.3f}",
            flush=True,
        )

    if failures:
        detail = "; ".join(f"{item['record']}: {item['reason']}" for item in failures[:5])
        raise SystemExit(
            "Derivation aborted because development records failed to load. "
            "Failing closed prevents a silently selected subset. " + detail
        )
    if not diagnostic_subset and len(caches) != len(DEVELOPMENT_RECORDS):
        raise SystemExit(
            f"Expected {len(DEVELOPMENT_RECORDS)} usable development records, got {len(caches)}."
        )

    aggregate_results = [
        score_grid_point(caches, v2_gate, width, args.tolerance_ms)
        for v2_gate in v2_grid
        for width in width_grid
    ]
    selected_aggregate = max(
        aggregate_results,
        key=lambda item: (
            item["mean_f1"],
            item["mean_sensitivity"],
            -item["v2_gate_confidence"],
            -item["width_override_confidence"],
        ),
    )
    selected = score_grid_point(
        caches,
        selected_aggregate["v2_gate_confidence"],
        selected_aggregate["width_override_confidence"],
        args.tolerance_ms,
        include_records=True,
    )
    non_identifying = find_non_identifying_dimensions(
        aggregate_results,
        v2_grid,
        width_grid,
    )
    engagement = engagement_summary(
        caches,
        selected["v2_gate_confidence"],
        selected["width_override_confidence"],
    )

    v2_status = _parameter_status(
        name="v2_gate_confidence",
        selected=selected["v2_gate_confidence"],
        historical=DEFAULT_V2_GATE_CONFIDENCE,
        non_identifying=non_identifying,
        changed_records=engagement["records_where_v2_gate_changes_final_polarity"],
    )
    width_status = _parameter_status(
        name="width_override_confidence",
        selected=selected["width_override_confidence"],
        historical=DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
        non_identifying=non_identifying,
        changed_records=engagement["records_where_width_override_changes_final_polarity"],
    )
    recommended_thresholds = {
        "v2_gate_confidence": v2_status["recommended_frozen_value"],
        "width_override_confidence": width_status["recommended_frozen_value"],
    }

    report = {
        "schema": "electrotrace.mitdb_polarity_thresholds_extended_derivation/v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": selection_status,
        "freeze_eligible": freeze_eligible,
        "git_head": git_head(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "input_hashes": input_hashes,
        "implementation_hashes": implementation_hashes(),
        "protocol": {
            "objective": "mean record-level Stage-1 F1",
            "development_pool_records": [cache.record for cache in caches],
            "development_pool_size": len(caches),
            "locked_heldout_records_excluded": sorted(LOCKED_HELDOUT_RECORDS),
            "locked_heldout_labels_used": False,
            "incart_used": False,
            "incart_reachable": False,
            "diagnostic_subset": diagnostic_subset,
            "tolerance_ms": args.tolerance_ms,
            "v2_gate_grid": v2_grid,
            "width_override_grid": width_grid,
            "selection_rule": (
                "maximize mean record-level Stage-1 F1; sensitivity, then lower v2 gate, "
                "then lower width gate are deterministic tie-breaks only"
            ),
        },
        "evaluation_integrity": {
            "threshold_selection_uses_locked_heldout_labels": False,
            "threshold_selection_uses_incart_labels": False,
            "adaptive_polarity_mechanism_was_historically_informed_by_locked_data": True,
            "width_override_mechanism_was_historically_informed_by_incart": True,
            "heldout_interpretation_after_freeze": (
                "legacy non-regression only; threshold re-derivation does not restore prospective "
                "held-out status for the already-designed adaptive-polarity mechanism"
            ),
            "prospective_validation_requires": (
                "an independent dataset not used to design or tune the adaptive-polarity mechanism, including neither MIT-BIH nor INCART"
            ),
        },
        "historical_defaults": {
            "v2_gate_confidence": DEFAULT_V2_GATE_CONFIDENCE,
            "width_override_confidence": DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
        },
        "selected_result": selected,
        "non_identifying_dimensions": non_identifying,
        "engagement_summary": engagement,
        "parameter_status": {
            "v2_gate_confidence": v2_status,
            "width_override_confidence": width_status,
        },
        "recommended_thresholds": recommended_thresholds,
        "per_record_raw_confidence": {
            cache.record: {
                "raw_confidence": cache.raw_confidence,
                "ratio_polarity": cache.ratio_polarity,
                "v2_polarity": cache.v2_polarity,
                "v2_confidence": cache.v2_confidence,
                "width_preferred": cache.width_preferred,
            }
            for cache in caches
        },
        "all_results": aggregate_results,
        "freeze_instruction": (
            "Use recommended_thresholds, not selected_result, for frozen evaluation. "
            "The locked evaluator must verify this report before accepting a non-default v2 gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n===== RESULT =====")
    print(f"development records: {len(caches)}")
    print(f"selected grid point: {selected['v2_gate_confidence']:.3f}, {selected['width_override_confidence']:.3f}")
    print("non-identifying dimensions:", non_identifying or "none")
    print("recommended frozen thresholds:", json.dumps(recommended_thresholds, sort_keys=True))
    print(
        "Interpretation: threshold selection excludes locked/INCART labels, but adaptive-polarity "
        "held-out results remain legacy non-regression because the mechanism itself was historically "
        "informed by MIT-BIH record 207/pooled behavior."
    )
    print("Written:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
