#!/usr/bin/env python3
"""Post-hoc EDB failure forensics.

EDB is exposed development data after the archived one-shot prospective run.
This script diagnoses the low-F1 tail without changing the frozen model,
threshold, polarity gates, or archived first-run result.

Channel 1 is evaluated only as a post-hoc lead-rescue diagnostic. Its metrics
must never replace the preregistered channel-0 prospective result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features
from electrotrace.fp_analysis import analyze_record
from electrotrace.scale_estimation import estimate_stage1_scale
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, match_peaks
from electrotrace.validation_detectors import (
    _candidate_set,
    detect_r_peaks_two_stage,
    select_signal_polarity,
)
from electrotrace.wfdb_records import load_annotated_record

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUTOFF = 0.80


def load_polarity_threshold_report(path: Path):
    """Load the sibling MIT-BIH verifier in both import and direct-exec modes."""
    root = str(REPO_ROOT)
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        from scripts.evaluate_frozen_model_mitdb import (
            _load_polarity_threshold_report as loader,
        )
        return loader(path)
    finally:
        if added:
            sys.path.remove(root)



def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_first_run(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "electrotrace.external_edb_prospective_validation/v1":
        raise SystemExit("unexpected first-run EDB schema")
    if report.get("evidence_status") != "prospective_external_evaluation_first_run":
        raise SystemExit("input is not the archived first prospective EDB run")
    integrity = report.get("evaluation_integrity", {})
    if integrity.get("records_scored") != 90 or integrity.get("records_skipped") != 0:
        raise SystemExit("archived EDB report must contain the complete 90-record cohort")
    if integrity.get("retraining") is not False:
        raise SystemExit("archived EDB report unexpectedly indicates retraining")
    return report


def verify_semantic_model_identity(
    model: CandidateSuppressor,
    first_run: dict,
) -> dict:
    """Verify model semantics while treating skops bytes as serialization detail.

    The trusted legacy pickle is checksum-gated and prediction-equivalence-tested
    by the workflow before this script runs. A fresh skops dump may have different
    bytes from the archived first-run dump, so byte identity is recorded but is
    not a scientific identity criterion.
    """
    archived_model = first_run.get("model", {})
    archived_metadata = archived_model.get("metadata")
    if not isinstance(archived_metadata, dict):
        raise SystemExit("archived first-run report is missing model metadata")

    current_metadata = model.metadata.to_dict()
    if current_metadata != archived_metadata:
        mismatches = {
            key: {
                "archived": archived_metadata.get(key),
                "current": current_metadata.get(key),
            }
            for key in sorted(set(archived_metadata) | set(current_metadata))
            if archived_metadata.get(key) != current_metadata.get(key)
        }
        raise SystemExit(
            "model semantic metadata does not match archived prospective run: "
            + json.dumps(mismatches, sort_keys=True)
        )

    if not model.fitted:
        raise SystemExit("reconstructed model is not fitted")

    return {
        "semantic_metadata_match": True,
        "archived_metadata": archived_metadata,
        "archived_serialized_model_sha256": archived_model.get("sha256"),
        "archived_serialized_sidecar_sha256": archived_model.get("sidecar_sha256"),
    }


def select_low_tail(first_run: dict, cutoff: float) -> list[dict]:
    rows = [
        row
        for row in first_run["record_results"]
        if float(row["f1"]) < float(cutoff)
    ]
    return sorted(rows, key=lambda row: (float(row["f1"]), str(row["record"])))


def _quantiles(values: np.ndarray) -> dict | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n": int(values.size),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def _feature_medians(features: np.ndarray, names: list[str], mask: np.ndarray) -> dict:
    wanted = ("amplitude_z", "prominence_z", "width_s", "qrs_band_fraction", "local_rms")
    out: dict[str, float | None] = {}
    for name in wanted:
        idx = names.index(name)
        vals = features[mask, idx] if features.size and mask.size else np.asarray([])
        out[name] = float(np.median(vals)) if vals.size else None
    return out


def diagnose_channel(
    base: Path,
    *,
    channel: int,
    model: CandidateSuppressor,
    v2_gate: float,
    width_gate: float,
    scale_method: str,
    tolerance_ms: float,
) -> dict:
    annotated = load_annotated_record(
        base,
        channel=channel,
        extension="atr",
        beat_symbols=DEFAULT_BEAT_SYMBOLS,
        tolerance_ms=tolerance_ms,
        policy="error",
    )
    if annotated.signal is None:
        raise RuntimeError("signal unexpectedly not loaded")
    signal = annotated.signal
    fs_hz = float(annotated.fs_hz)
    reference = annotated.reference

    decision = select_signal_polarity(
        signal,
        fs_hz,
        scale_method=scale_method,
        v2_gate_confidence=v2_gate,
        width_override_confidence=width_gate,
    )
    centered = signal - np.median(signal)
    scale = estimate_stage1_scale(centered, fs_hz, method=scale_method)
    stream = centered if decision.polarity == "positive" else -centered
    candidates, prominences = _candidate_set(stream, fs_hz, scale)

    if candidates.size:
        features, names = _candidate_features(
            signal,
            fs_hz,
            candidates,
            prominences,
            scale_method=scale_method,
        )
        probabilities = model.predict_proba(features)
    else:
        features = np.empty((0, 0), dtype=float)
        names = []
        probabilities = np.empty(0, dtype=float)

    threshold = float(model.metadata.threshold)
    forensic = analyze_record(
        signal,
        fs_hz,
        candidates,
        probabilities,
        reference,
        threshold=threshold,
        tolerance_ms=tolerance_ms,
    )
    cand_summary = forensic["candidates"]

    # The final detector output must come from the exact production path used by
    # the archived prospective run. The hand-built all-candidate table above is
    # diagnostic only: it is useful for AUC/feature analysis but must never
    # substitute for detect_r_peaks_two_stage() when reproducing final metrics.
    retained_samples, retained_probabilities = detect_r_peaks_two_stage(
        signal,
        fs_hz,
        model,
        polarity="adaptive",
        recovery=False,
        scale_method=scale_method,
        width_override_confidence=width_gate,
        v2_gate_confidence=v2_gate,
    )
    metrics = match_peaks(
        retained_samples,
        reference,
        fs_hz,
        tolerance_ms=tolerance_ms,
    ).to_dict()

    # Canonical retained samples should belong to the Stage-1 stream reconstructed
    # for diagnostics. Keep this explicit so a future implementation divergence
    # fails loudly instead of producing misleading feature summaries.
    if retained_samples.size and not np.all(np.isin(retained_samples, candidates)):
        raise RuntimeError(
            "canonical two-stage output contains samples absent from diagnostic Stage-1 candidates"
        )
    retained_mask = np.isin(candidates, retained_samples)
    ref_n = int(len(reference))
    stage1_n = int(len(candidates))
    retained_n = int(len(retained_samples))

    rr = np.diff(reference) / fs_hz if len(reference) > 1 else np.asarray([])
    candidate_reference_coverage = (
        float(cand_summary["n_true_candidates"] / ref_n) if ref_n else None
    )
    feature_medians_all = (
        _feature_medians(features, names, np.ones(stage1_n, dtype=bool))
        if stage1_n
        else {}
    )
    feature_medians_retained = (
        _feature_medians(features, names, retained_mask)
        if stage1_n
        else {}
    )
    retained_probabilities = probabilities[retained_mask]
    retained_samples = candidates[retained_mask]
    retained_rr = (
        np.diff(retained_samples) / fs_hz
        if retained_samples.size > 1
        else np.asarray([], dtype=float)
    )
    retained_rr_median = float(np.median(retained_rr)) if retained_rr.size else None
    retained_rr_mad = (
        float(np.median(np.abs(retained_rr - np.median(retained_rr))))
        if retained_rr.size
        else None
    )
    retained_rr_mad_fraction = (
        float(retained_rr_mad / retained_rr_median)
        if retained_rr_median is not None and retained_rr_median > 0
        else None
    )

    return {
        "channel": int(channel),
        "fs_hz": fs_hz,
        "reference_count": ref_n,
        "selected_polarity": decision.polarity,
        "polarity_confidence": float(decision.confidence),
        "positive_candidates_for_polarity_rule": int(decision.positive_candidates),
        "negative_candidates_for_polarity_rule": int(decision.negative_candidates),
        "stage1_candidates": stage1_n,
        "stage1_candidates_over_reference": float(stage1_n / ref_n) if ref_n else None,
        "candidate_reference_coverage": candidate_reference_coverage,
        "stage2_retained": retained_n,
        "stage2_retained_over_reference": float(retained_n / ref_n) if ref_n else None,
        "suppression_rate": float(1.0 - retained_n / stage1_n) if stage1_n else 0.0,
        "operating_threshold": threshold,
        "label_free_quality": {
            "duration_s": float(signal.size / fs_hz),
            "retention_fraction": float(retained_n / stage1_n) if stage1_n else 0.0,
            "retained_probability": _quantiles(retained_probabilities),
            "retained_rr_s": _quantiles(retained_rr),
            "retained_rr_median_s": retained_rr_median,
            "retained_rr_mad_s": retained_rr_mad,
            "retained_rr_mad_fraction": retained_rr_mad_fraction,
            "retained_rate_bpm": (
                float(retained_n / (signal.size / fs_hz) * 60.0)
                if signal.size
                else None
            ),
        },
        "metrics": metrics,
        "stage2_candidate_scoring": cand_summary,
        "diagnostic_manual_threshold_metrics": forensic["metrics"],
        "manual_threshold_matches_canonical": bool(
            np.array_equal(candidates[probabilities >= threshold], retained_samples)
        ),
        "false_positive_categories": forensic["false_positive_categories"],
        "signal": {
            "stage1_scale": float(scale),
            "std": float(np.std(centered)),
            "mad": float(np.median(np.abs(centered - np.median(centered)))),
            "p05": float(np.percentile(centered, 5)),
            "p95": float(np.percentile(centered, 95)),
            "reference_rr_s": _quantiles(rr),
            "reference_median_bpm": (
                float(60.0 / np.median(rr)) if rr.size and np.median(rr) > 0 else None
            ),
            "candidate_prominence_over_scale": _quantiles(
                prominences / max(float(scale), 1e-12)
            ),
        },
        "feature_medians_all_candidates": feature_medians_all,
        "feature_medians_retained": feature_medians_retained,
    }


def classify_primary(primary: dict, *, cutoff: float) -> str:
    m = primary["metrics"]
    if float(m["f1"]) >= cutoff:
        return "not_low_tail"
    stage1_ratio = float(primary["stage1_candidates_over_reference"] or 0.0)
    coverage = float(primary["candidate_reference_coverage"] or 0.0)
    suppression = float(primary["suppression_rate"])
    sensitivity = float(m["sensitivity"])
    ppv = float(m["positive_predictive_value"])

    if stage1_ratio < 0.5:
        return "candidate_starvation"
    if coverage < 0.5:
        return "candidate_misalignment_or_wrong_deflections"
    if coverage >= 0.8 and suppression > 0.7 and sensitivity < 0.8:
        return "stage2_over_suppression"
    if ppv < 0.5 and sensitivity >= 0.3:
        return "over_detection_or_mixed_fp"
    return "mixed_or_other"


def build_row(record: str, archived: dict, primary: dict, alternate: dict, cutoff: float) -> dict:
    mode = classify_primary(primary, cutoff=cutoff)
    p = primary["metrics"]
    a = alternate["metrics"]
    return {
        "record": record,
        "archived_f1": float(archived["f1"]),
        "channel0_f1": float(p["f1"]),
        "channel0_sensitivity": float(p["sensitivity"]),
        "channel0_ppv": float(p["positive_predictive_value"]),
        "channel0_stage1_over_ref": primary["stage1_candidates_over_reference"],
        "channel0_candidate_reference_coverage": primary["candidate_reference_coverage"],
        "channel0_suppression_rate": primary["suppression_rate"],
        "channel0_candidate_auc": primary["stage2_candidate_scoring"]["auc"],
        "channel0_oracle_threshold_f1": primary["stage2_candidate_scoring"]["oracle_best_f1"],
        "channel1_f1_posthoc": float(a["f1"]),
        "channel1_sensitivity_posthoc": float(a["sensitivity"]),
        "channel1_ppv_posthoc": float(a["positive_predictive_value"]),
        "channel1_minus_channel0_f1": float(a["f1"] - p["f1"]),
        "heuristic_failure_mode": mode,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edb-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--first-run-report", type=Path, required=True)
    ap.add_argument("--polarity-threshold-report", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--f1-cutoff", type=float, default=DEFAULT_CUTOFF)
    args = ap.parse_args()

    if not 0.0 < args.f1_cutoff <= 1.0:
        raise SystemExit("--f1-cutoff must be in (0, 1]")

    first_run = load_first_run(args.first_run_report)
    low_tail = select_low_tail(first_run, args.f1_cutoff)
    if not low_tail:
        raise SystemExit("no low-tail records selected")

    model = CandidateSuppressor.load(args.model)
    model_identity = verify_semantic_model_identity(model, first_run)
    current_model_sha = sha256_file(args.model)
    current_sidecar = Path(str(args.model) + ".json")
    if not current_sidecar.is_file():
        raise SystemExit(f"missing reconstructed model sidecar: {current_sidecar}")
    current_sidecar_sha = sha256_file(current_sidecar)

    derivation, width_gate, v2_gate = load_polarity_threshold_report(
        args.polarity_threshold_report
    )
    archived_thresholds = first_run["polarity_threshold_derivation"]["recommended_thresholds"]
    if float(v2_gate) != float(archived_thresholds["v2_gate_confidence"]):
        raise SystemExit("v2 gate does not match archived first-run gate")
    if float(width_gate) != float(archived_thresholds["width_override_confidence"]):
        raise SystemExit("width gate does not match archived first-run gate")

    protocol = first_run["protocol"]["detector"]
    scale_method = str(protocol["stage1_scale_method"])
    tolerance_ms = float(protocol["tolerance_ms"])
    records = []
    csv_rows = []
    for i, archived in enumerate(low_tail, start=1):
        record = str(archived["record"])
        base = args.edb_dir / record
        primary = diagnose_channel(
            base,
            channel=0,
            model=model,
            v2_gate=v2_gate,
            width_gate=width_gate,
            scale_method=scale_method,
            tolerance_ms=tolerance_ms,
        )
        if abs(float(primary["metrics"]["f1"]) - float(archived["f1"])) > 1e-12:
            raise SystemExit(
                f"{record}: channel-0 diagnostic does not reproduce archived F1 "
                f"({primary['metrics']['f1']} != {archived['f1']})"
            )
        alternate = diagnose_channel(
            base,
            channel=1,
            model=model,
            v2_gate=v2_gate,
            width_gate=width_gate,
            scale_method=scale_method,
            tolerance_ms=tolerance_ms,
        )
        mode = classify_primary(primary, cutoff=args.f1_cutoff)
        row = build_row(record, archived, primary, alternate, args.f1_cutoff)
        records.append(
            {
                "record": record,
                "archived_prospective_metrics": {
                    k: archived[k]
                    for k in (
                        "sensitivity",
                        "positive_predictive_value",
                        "f1",
                        "true_positive",
                        "false_positive",
                        "false_negative",
                    )
                },
                "heuristic_failure_mode": mode,
                "channel0_reproduction": primary,
                "channel1_posthoc_diagnostic": alternate,
                "channel1_minus_channel0_f1": row["channel1_minus_channel0_f1"],
                "alternate_lead_is_posthoc_development_only": True,
            }
        )
        csv_rows.append(row)
        print(
            f"[{i}/{len(low_tail)}] {record}: mode={mode}, "
            f"ch0 F1={primary['metrics']['f1']:.4f}, "
            f"ch1(posthoc) F1={alternate['metrics']['f1']:.4f}"
        )

    counts = Counter(r["heuristic_failure_mode"] for r in records)
    alt_rescue = sorted(
        records,
        key=lambda r: r["channel1_minus_channel0_f1"],
        reverse=True,
    )

    incart_path = Path("validation_reports/incart_domain_shift_analysis_2026-09-12.json")
    incart_context = None
    if incart_path.is_file():
        incart = json.loads(incart_path.read_text(encoding="utf-8"))
        incart_context = {
            "source": str(incart_path),
            "failure_mode_counts": incart.get("failure_mode_counts"),
            "note": (
                "INCART and EDB taxonomies are both exposed-data development diagnostics. "
                "Category rules are related but not identical, so counts are contextual, "
                "not a head-to-head endpoint."
            ),
        }

    report = {
        "schema": "electrotrace.edb_posthoc_failure_analysis/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "posthoc_exposed_dataset_development_analysis",
        "git_head": git_head(),
        "inputs": {
            "first_run_report": str(args.first_run_report),
            "first_run_sha256": sha256_file(args.first_run_report),
            "model": str(args.model),
            "model_semantic_identity": model_identity,
            "reconstructed_model_sha256": current_model_sha,
            "reconstructed_sidecar_sha256": current_sidecar_sha,
            "serialized_model_sha_matches_archived": (
                current_model_sha == first_run["model"].get("sha256")
            ),
            "serialized_sidecar_sha_matches_archived": (
                current_sidecar_sha == first_run["model"].get("sidecar_sha256")
            ),
            "serialization_note": (
                "Fresh skops bytes may differ across equivalent reconstructions; "
                "semantic model identity is established by archived metadata plus "
                "checksum-gated trusted-pickle prediction equivalence in the workflow."
            ),
            "polarity_derivation": str(args.polarity_threshold_report),
            "polarity_derivation_sha256": sha256_file(args.polarity_threshold_report),
        },
        "guardrail": (
            "EDB is exposed. This report may guide development, but no EDB-motivated "
            "change may be called prospectively validated on EDB. Channel 1 metrics "
            "are post-hoc diagnostics only and do not replace the archived channel-0 result."
        ),
        "selection": {
            "criterion": "archived prospective record F1 below cutoff",
            "f1_cutoff": float(args.f1_cutoff),
            "n_selected": len(records),
            "records": [r["record"] for r in records],
        },
        "heuristic_failure_mode_counts": dict(sorted(counts.items())),
        "alternate_lead_rescue_ranking": [
            {
                "record": r["record"],
                "channel1_minus_channel0_f1": r["channel1_minus_channel0_f1"],
                "channel0_f1": r["channel0_reproduction"]["metrics"]["f1"],
                "channel1_f1_posthoc": r["channel1_posthoc_diagnostic"]["metrics"]["f1"],
            }
            for r in alt_rescue
        ],
        "incart_exposed_context": incart_context,
        "records": records,
        "interpretation_rules": {
            "candidate_starvation": "stage1_candidates/reference_count < 0.5",
            "candidate_misalignment_or_wrong_deflections": (
                "stage1 count is not starved but fewer than 50% of references have "
                "a matchable Stage-1 candidate within the fixed tolerance"
            ),
            "stage2_over_suppression": (
                "candidate reference coverage >= 0.8, suppression > 0.7, "
                "and final sensitivity < 0.8"
            ),
            "over_detection_or_mixed_fp": "final PPV < 0.5 with sensitivity >= 0.3",
            "mixed_or_other": "none of the above heuristic development categories",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nFailure modes:", dict(sorted(counts.items())))
    print("Wrote:", args.output_json)
    print("Wrote:", args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
