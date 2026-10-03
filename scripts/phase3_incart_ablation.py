#!/usr/bin/env python3
"""Phase-3 INCART ablation harness.

This script keeps the historical MIT-BIH-trained model fixed and evaluates:
1) raw INCART;
2) canonical-rate resampling;
3) robust per-record scaling;
4) resampling + scaling;
5) RF threshold sweeps on the frozen feature/model path.

Threshold results are exploratory operating curves. The script never chooses a
deployment threshold from INCART labels and never retrains on INCART.
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
import scipy.signal as sps
import wfdb

from electrotrace import __version__
from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, match_peaks
from electrotrace.wfdb_records import (
    POLICIES,
    RecordExcluded,
    load_annotated_record,
    summarize_audits,
)


@dataclass(frozen=True)
class Transform:
    name: str
    target_fs: float | None
    robust_scale: bool


TRANSFORMS = (
    Transform("raw", None, False),
    Transform("resample_360", 360.0, False),
    Transform("robust_scale", None, True),
    Transform("resample_360_robust_scale", 360.0, True),
)


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


def package_versions() -> dict:
    versions = {}
    for module_name in ("numpy", "scipy", "sklearn", "skops", "wfdb", "pandas"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "missing"
    return versions


def robust_scale(x: np.ndarray) -> np.ndarray:
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(x - center))
    if not np.isfinite(scale) or scale <= 1e-12:
        return x - center
    return (x - center) / scale


def transform_signal(
    signal: np.ndarray, fs_hz: float, transform: Transform
) -> tuple[np.ndarray, float]:
    x = np.asarray(signal, dtype=float)
    fs = float(fs_hz)
    if transform.target_fs is not None and not np.isclose(fs, transform.target_fs):
        target = float(transform.target_fs)
        from fractions import Fraction

        ratio = Fraction(target / fs).limit_denominator(1000)
        x = sps.resample_poly(x, ratio.numerator, ratio.denominator)
        fs = target
    if transform.robust_scale:
        x = robust_scale(x)
    return np.asarray(x, dtype=float), fs


def record_names(data_dir: Path) -> list[str]:
    records = sorted(path.stem for path in data_dir.glob("*.hea"))
    if records:
        return records
    names = [Path(x).name for x in wfdb.get_record_list("incartdb")]
    return sorted(names)


def score_candidates(
    model: CandidateSuppressor,
    signal: np.ndarray,
    fs_hz: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run Stage 1 + Stage 2 scoring once and return all Stage-2 candidates.

    Thresholding is deliberately deferred until after scoring so a threshold
    sweep reuses the same candidate probabilities instead of rerunning
    Stage 1 and feature extraction for every threshold.
    """
    from electrotrace.validation_detectors import detect_r_peaks, detect_r_peaks_two_stage

    stage1 = detect_r_peaks(
        signal, fs_hz, polarity="adaptive", scale_method="windowed_std"
    )
    candidates, probabilities = detect_r_peaks_two_stage(
        signal,
        fs_hz,
        model,
        polarity="adaptive",
        recovery=False,
        scale_method="windowed_std",
        threshold=0.0,
    )
    return candidates.astype(int), probabilities.astype(float), int(len(stage1))


def evaluate_scored(
    model: CandidateSuppressor,
    candidates: np.ndarray,
    probabilities: np.ndarray,
    stage1_count: int,
    fs_hz: float,
    refs: np.ndarray,
    *,
    threshold: float | None,
) -> dict:
    selected_threshold = float(model.metadata.threshold if threshold is None else threshold)
    detected = candidates[probabilities >= selected_threshold]
    metrics = match_peaks(detected, refs, fs_hz, tolerance_ms=75.0)
    return {
        "reference_count": int(len(refs)),
        "stage1_candidates": int(stage1_count),
        "detected_count": int(len(detected)),
        "sensitivity": metrics.sensitivity,
        "ppv": metrics.positive_predictive_value,
        "f1": metrics.f1,
        "median_abs_timing_error_ms": metrics.median_absolute_timing_error_ms,
        "p95_abs_timing_error_ms": metrics.p95_absolute_timing_error_ms,
        "threshold": selected_threshold,
        "probability_median": float(np.median(probabilities)) if len(probabilities) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incart-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--transforms",
        default=",".join(t.name for t in TRANSFORMS),
    )
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Defaults to every local INCART record.",
    )
    parser.add_argument(
        "--annotation-policy",
        choices=POLICIES,
        default="error",
        help=(
            "error (historical): records with invalid reference annotations are "
            "excluded. drop_edges: drop invalid edge annotations only when an "
            "independent detector confirms the remaining reference is aligned."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_reports/incart_phase3_ablation.json"),
    )
    args = parser.parse_args()

    model = CandidateSuppressor.load(args.model)
    selected = {x.strip() for x in args.transforms.split(",") if x.strip()}
    transforms = [t for t in TRANSFORMS if t.name in selected]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    names = args.records or record_names(args.incart_dir)

    results: list[dict] = []
    skipped: list[dict] = []
    audits = []

    for index, name in enumerate(names, start=1):
        try:
            annotated = load_annotated_record(
                args.incart_dir / name,
                beat_symbols=DEFAULT_BEAT_SYMBOLS,
                policy=args.annotation_policy,
            )
            audits.append(annotated.audit)
            signal, fs_hz, refs = annotated.signal, annotated.fs_hz, annotated.reference
            for transform in transforms:
                x, xfs = transform_signal(signal, fs_hz, transform)
                scale = xfs / fs_hz
                xrefs = np.rint(refs * scale).astype(int)
                candidates, probabilities, stage1_count = score_candidates(
                    model, x, xfs
                )
                baseline = evaluate_scored(
                    model,
                    candidates,
                    probabilities,
                    stage1_count,
                    xfs,
                    xrefs,
                    threshold=None,
                )
                baseline.update(
                    {
                        "record": name,
                        "transform": transform.name,
                        "original_fs_hz": fs_hz,
                        "evaluated_fs_hz": xfs,
                    }
                )
                results.append(baseline)

                for threshold in thresholds:
                    sweep = evaluate_scored(
                        model,
                        candidates,
                        probabilities,
                        stage1_count,
                        xfs,
                        xrefs,
                        threshold=threshold,
                    )
                    sweep.update(
                        {
                            "record": name,
                            "transform": transform.name,
                            "original_fs_hz": fs_hz,
                            "evaluated_fs_hz": xfs,
                        }
                    )
                    results.append(sweep)

            print(f"[{index}/{len(names)}] {name}: done", flush=True)
        except RecordExcluded as exc:
            audits.append(exc.audit)
            skipped.append({"record": name, "reason": exc.reason})
            print(f"[{index}/{len(names)}] {name}: EXCLUDED: {exc.reason}", flush=True)
        except Exception as exc:
            skipped.append({"record": name, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(names)}] {name}: SKIPPED: {exc}", flush=True)

    def summarize(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            "mean_f1": float(np.mean([r["f1"] for r in rows])),
            "mean_sensitivity": float(np.mean([r["sensitivity"] for r in rows])),
            "mean_ppv": float(np.mean([r["ppv"] for r in rows])),
            "median_abs_timing_error_ms": float(
                np.median(
                    [r["median_abs_timing_error_ms"] for r in rows if r["median_abs_timing_error_ms"] is not None]
                )
            )
            if any(r["median_abs_timing_error_ms"] is not None for r in rows)
            else None,
            "mean_stage1_candidates": float(np.mean([r["stage1_candidates"] for r in rows])),
        }

    baseline_rows = [
        r for r in results if r["threshold"] == float(model.metadata.threshold)
    ]
    threshold_rows = [r for r in results if r["threshold"] != float(model.metadata.threshold)]

    summary_by_transform = {
        transform.name: summarize([r for r in baseline_rows if r["transform"] == transform.name])
        for transform in transforms
    }
    summary_by_threshold = {
        transform.name: {
            str(threshold): summarize(
                [
                    r
                    for r in threshold_rows
                    if r["transform"] == transform.name
                    and np.isclose(r["threshold"], threshold)
                ]
            )
            for threshold in thresholds
        }
        for transform in transforms
    }

    input_hashes = {}
    for record_name in names:
        for source in sorted(args.incart_dir.glob(f"{record_name}.*")):
            if source.is_file():
                input_hashes[f"{record_name}::{source.name}"] = sha256_file(source)

    report = {
        "schema": "electrotrace.incart_phase3_ablation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "Virelion-Biotech/Virelion-ElectroTrace",
        "git_head": git_head(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "model_metadata": model.metadata.to_dict(),
        "input_hashes": input_hashes,
        "protocol": {
            "dataset": "PhysioNet INCART",
            "records_requested": names,
            "tolerance_ms": 75.0,
            "transforms": [t.name for t in transforms],
            "thresholds": thresholds,
            "threshold_selection_note": "Threshold sweep is exploratory; no threshold is selected from INCART labels.",
            "retraining": False,
            "annotation_policy": args.annotation_policy,
        },
        "annotation_audit": {
            "summary": summarize_audits(audits),
            "records": [a.to_dict() for a in audits],
        },
        "summary_by_transform": summary_by_transform,
        "summary_by_threshold": summary_by_threshold,
        "record_results": results,
        "skipped_records": skipped,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n===== BASELINE TRANSFORM SUMMARIES =====")
    for name, value in summary_by_transform.items():
        print(name, json.dumps(value, sort_keys=True))
    print("\nWritten:", args.output)
    print("Skipped:", len(skipped))
    return 0 if not skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())
