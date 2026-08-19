"""Publication-oriented validation summaries built on the core detector metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import t

from .provenance import DatasetManifest
from .validation import RecordValidation

VALIDATION_SCHEMA_VERSION = "electrotrace-validation-report-v1"


def _finite(values: Sequence[float | int | None]) -> np.ndarray:
    return np.asarray([float(x) for x in values if x is not None and np.isfinite(x)], dtype=float)


def mean_ci95(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    """Return a two-sided 95% t-based confidence interval for a sample mean."""
    x = _finite(values)
    if x.size == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(x))
    if x.size == 1:
        return {"n": 1, "mean": mean, "sd": None, "ci95_low": None, "ci95_high": None}
    sd = float(np.std(x, ddof=1))
    critical = float(t.ppf(0.975, df=x.size - 1))
    margin = critical * sd / np.sqrt(x.size)
    return {"n": int(x.size), "mean": mean, "sd": sd, "ci95_low": mean - margin, "ci95_high": mean + margin}


def bootstrap_record_ci(results: Sequence[RecordValidation], metric: str, n_bootstrap: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Bootstrap a record-level metric to preserve the experimental unit."""
    if not results:
        raise ValueError("At least one record result is required")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    values = _finite([getattr(r.metrics, metric) for r in results])
    if len(values) != len(results):
        raise ValueError(f"Metric '{metric}' is unavailable for at least one record")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boot = np.mean(values[samples], axis=1)
    return {
        "unit": "record",
        "n_records": len(results),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "mean": float(np.mean(values)),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
    }


def summarize_records_rigorous(results: Sequence[RecordValidation], n_bootstrap: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Produce pooled and macro record-level metrics with uncertainty estimates."""
    if not results:
        raise ValueError("At least one validation result is required")
    pooled_tp = sum(r.metrics.true_positive for r in results)
    pooled_fp = sum(r.metrics.false_positive for r in results)
    pooled_fn = sum(r.metrics.false_negative for r in results)
    pooled_ref = sum(r.metrics.reference_count for r in results)
    pooled_det = sum(r.metrics.detected_count for r in results)
    pooled_sens = pooled_tp / pooled_ref if pooled_ref else 0.0
    pooled_ppv = pooled_tp / pooled_det if pooled_det else 0.0
    pooled_f1 = 2 * pooled_sens * pooled_ppv / (pooled_sens + pooled_ppv) if pooled_sens + pooled_ppv else 0.0

    macro = {
        "sensitivity": mean_ci95([r.metrics.sensitivity for r in results]),
        "positive_predictive_value": mean_ci95([r.metrics.positive_predictive_value for r in results]),
        "f1": mean_ci95([r.metrics.f1 for r in results]),
        "mean_timing_error_ms": mean_ci95([r.metrics.mean_timing_error_ms for r in results]),
        "median_timing_error_ms": mean_ci95([r.metrics.median_timing_error_ms for r in results]),
        "mean_absolute_timing_error_ms": mean_ci95([r.metrics.mean_absolute_timing_error_ms for r in results]),
        "median_absolute_timing_error_ms": mean_ci95([r.metrics.median_absolute_timing_error_ms for r in results]),
        "p95_absolute_timing_error_ms": mean_ci95([r.metrics.p95_absolute_timing_error_ms for r in results]),
        "max_absolute_timing_error_ms": mean_ci95([r.metrics.max_absolute_timing_error_ms for r in results]),
    }
    bootstrap = {
        "sensitivity": bootstrap_record_ci(results, "sensitivity", n_bootstrap=n_bootstrap, seed=seed),
        "positive_predictive_value": bootstrap_record_ci(results, "positive_predictive_value", n_bootstrap=n_bootstrap, seed=seed + 1),
        "f1": bootstrap_record_ci(results, "f1", n_bootstrap=n_bootstrap, seed=seed + 2),
    }
    return {
        "records": len(results),
        "pooled": {
            "reference_count": int(pooled_ref),
            "detected_count": int(pooled_det),
            "true_positive": int(pooled_tp),
            "false_positive": int(pooled_fp),
            "false_negative": int(pooled_fn),
            "sensitivity": float(pooled_sens),
            "positive_predictive_value": float(pooled_ppv),
            "f1": float(pooled_f1),
        },
        "macro_record": macro,
        "bootstrap_record": bootstrap,
        "per_record": [r.to_dict() for r in results],
    }


def build_validation_report(
    manifest: DatasetManifest,
    results: Sequence[RecordValidation],
    *,
    detector_name: str,
    detector_parameters: dict[str, Any],
    annotation_extension: str,
    beat_symbols: Sequence[str] | None,
    tolerance_ms: float,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a self-contained, hashable validation report."""
    manifest.validate()
    result_records = tuple(r.record for r in results)
    if set(result_records) - set(manifest.records):
        raise ValueError("Validation results contain records absent from the dataset manifest")
    if len(result_records) != len(set(result_records)):
        raise ValueError("Validation results must contain at most one result per record")
    report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "manifest": manifest.to_dict(),
        "manifest_sha256": manifest.sha256(),
        "detector": {"name": detector_name, "parameters": detector_parameters},
        "reference": {
            "annotation_extension": annotation_extension,
            "beat_symbols": list(beat_symbols) if beat_symbols is not None else None,
            "matching_tolerance_ms": float(tolerance_ms),
        },
        "summary": summarize_records_rigorous(results, n_bootstrap=n_bootstrap, seed=seed),
        "failures": sorted(set(manifest.records) - set(result_records)),
    }
    return report


def write_validation_report(report: dict[str, Any], output: str | Path) -> Path:
    """Write a deterministic, pretty-printed JSON validation report."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return path
