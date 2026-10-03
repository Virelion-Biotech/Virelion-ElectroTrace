#!/usr/bin/env python3
"""Post-hoc label-free feature audit for exposed LTAFDB.

This analysis is development-only. It computes lead-quality summaries from
signal/model outputs, then compares them with exposed reference outcomes.
It must not alter or relabel the immutable prospective selector result.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features
from electrotrace.scale_estimation import estimate_stage1_scale
from electrotrace.validation import match_peaks
from electrotrace.validation_detectors import (
    _candidate_set,
    detect_r_peaks_two_stage,
    select_signal_polarity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_prospective = importlib.import_module("scripts.evaluate_ltafdb_lead_selector_prospective")
_audit = importlib.import_module("scripts.analyze_ltafdb_lead_selector_posthoc")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _quantiles(values: np.ndarray) -> dict | None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        "n": int(x.size),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def _feature_medians(features: np.ndarray, names: list[str], mask: np.ndarray) -> dict:
    wanted = ("amplitude_z", "prominence_z", "width_s", "qrs_band_fraction", "local_rms")
    out = {}
    for name in wanted:
        idx = names.index(name)
        vals = features[mask, idx] if features.size and mask.size else np.asarray([])
        out[name] = float(np.median(vals)) if vals.size else None
    return out


def channel_quality(
    signal: np.ndarray,
    fs_hz: float,
    model: CandidateSuppressor,
    *,
    v2_gate: float,
    width_gate: float,
    scale_method: str,
) -> tuple[np.ndarray, dict]:
    """Return canonical detections and label-free lead-quality summaries."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("signal must be finite one-dimensional data")

    polarity = select_signal_polarity(
        x,
        fs_hz,
        scale_method=scale_method,
        v2_gate_confidence=v2_gate,
        width_override_confidence=width_gate,
    )
    centered = x - np.median(x)
    scale = estimate_stage1_scale(centered, fs_hz, method=scale_method)
    stream = centered if polarity.polarity == "positive" else -centered
    candidates, prominences = _candidate_set(stream, fs_hz, scale)

    if candidates.size:
        features, names = _candidate_features(
            x, fs_hz, candidates, prominences, scale_method=scale_method
        )
    else:
        features = np.empty((0, 0), dtype=float)
        names = []

    retained, retained_probabilities = detect_r_peaks_two_stage(
        x,
        fs_hz,
        model,
        polarity="adaptive",
        recovery=False,
        scale_method=scale_method,
        width_override_confidence=width_gate,
        v2_gate_confidence=v2_gate,
        threshold=float(model.metadata.threshold),
    )
    retained = np.asarray(retained, dtype=int)
    retained_probabilities = np.asarray(retained_probabilities, dtype=float)
    if retained.size != retained_probabilities.size:
        raise RuntimeError("retained samples/probabilities length mismatch")
    if retained.size and not np.all(np.isin(retained, candidates)):
        raise RuntimeError("canonical retained samples are absent from diagnostic candidates")

    retained_mask = np.isin(candidates, retained)
    rr = np.diff(retained) / fs_hz if retained.size > 1 else np.asarray([], dtype=float)
    rr_median = float(np.median(rr)) if rr.size else None
    rr_mad = (
        float(np.median(np.abs(rr - np.median(rr)))) if rr.size else None
    )
    rr_mad_fraction = (
        float(rr_mad / rr_median)
        if rr_median is not None and rr_median > 0 and rr_mad is not None
        else None
    )
    duration_s = float(x.size / fs_hz)

    return retained, {
        "selected_polarity": polarity.polarity,
        "polarity_confidence": float(polarity.confidence),
        "stage1_scale": float(scale),
        "signal_std": float(np.std(centered)),
        "signal_mad": float(np.median(np.abs(centered - np.median(centered)))),
        "stage1_candidates": int(candidates.size),
        "retained_count": int(retained.size),
        "retention_fraction": (
            float(retained.size / candidates.size) if candidates.size else 0.0
        ),
        "retained_rate_bpm": float(retained.size / duration_s * 60.0),
        "retained_probability": _quantiles(retained_probabilities),
        "retained_rr_s": _quantiles(rr),
        "retained_rr_median_s": rr_median,
        "retained_rr_mad_s": rr_mad,
        "retained_rr_mad_fraction": rr_mad_fraction,
        "candidate_prominence_over_scale": _quantiles(
            prominences / max(float(scale), 1e-12)
        ),
        "feature_medians_all_candidates": (
            _feature_medians(features, names, np.ones(candidates.size, dtype=bool))
            if candidates.size else {}
        ),
        "feature_medians_retained": (
            _feature_medians(features, names, retained_mask)
            if candidates.size else {}
        ),
    }


def _f1_relation(ch0_f1: float, ch1_f1: float) -> str:
    delta = float(ch1_f1) - float(ch0_f1)
    if delta > 1e-12:
        return "channel1_better"
    if delta < -1e-12:
        return "channel0_better"
    return "tie"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ltafdb-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--polarity-threshold-report", type=Path, required=True)
    ap.add_argument("--first-run-report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    first = _audit.load_first_run(args.first_run_report)
    protocol = _prospective.load_locked_protocol()
    records = list(protocol["dataset"]["records"])
    if [r["record"] for r in first["record_results"]] != records:
        raise SystemExit("archived first-run order differs from locked protocol")

    model = CandidateSuppressor.load(args.model)
    _prospective.verify_model(model, protocol)
    derivation, width_gate, v2_gate = _prospective._load_polarity_threshold_report(
        args.polarity_threshold_report
    )
    detector = protocol["detector"]
    if derivation.get("schema") != detector["polarity_threshold_report_schema"]:
        raise SystemExit("polarity derivation schema mismatch")
    if not _audit._same_number(v2_gate, detector["expected_v2_gate_confidence"]):
        raise SystemExit("v2 gate mismatch")
    if not _audit._same_number(width_gate, detector["expected_width_override_confidence"]):
        raise SystemExit("width gate mismatch")

    archived_by = {str(r["record"]): r for r in first["record_results"]}
    window = protocol["dataset"]["window"]
    start = int(window["start_sample"])
    stop = int(window["stop_sample_exclusive"])
    fs = float(protocol["dataset"]["expected_sampling_frequency_hz"])
    scale_method = str(detector["stage1_scale_method"])
    tolerance_ms = float(detector["tolerance_ms"])
    beat_symbols = frozenset(detector["beat_symbols"])

    rows = []
    for i, record in enumerate(records, start=1):
        rec = wfdb.rdrecord(
            str(args.ltafdb_dir / record),
            sampfrom=start,
            sampto=stop,
            channels=[0, 1],
            physical=True,
        )
        signals = np.asarray(rec.p_signal, dtype=float)
        if not _audit._same_number(rec.fs, fs):
            raise SystemExit(f"{record}: unexpected sampling frequency {rec.fs}")
        if signals.shape != (stop - start, 2) or not np.isfinite(signals).all():
            raise SystemExit(f"{record}: invalid frozen signal window {signals.shape}")

        detections = []
        qualities = []
        for ch in (0, 1):
            retained, quality = channel_quality(
                signals[:, ch],
                fs,
                model,
                v2_gate=v2_gate,
                width_gate=width_gate,
                scale_method=scale_method,
            )
            detections.append(retained)
            qualities.append(quality)

        annotation = wfdb.rdann(
            str(args.ltafdb_dir / record),
            str(detector["annotation_extension"]),
            sampfrom=start,
            sampto=stop,
        )
        reference, _ = _prospective._reference_from_annotation(
            annotation, beat_symbols, stop - start
        )
        metrics = [
            match_peaks(detections[ch], reference, fs, tolerance_ms=tolerance_ms)
            for ch in (0, 1)
        ]
        archived = archived_by[record]
        selected = int(archived["selected_channel"])
        _audit._assert_reproduces_archived(
            record,
            archived,
            selected,
            [
                qualities[0]["retained_probability"]["p50"]
                if qualities[0]["retained_probability"] else 0.0,
                qualities[1]["retained_probability"]["p50"]
                if qualities[1]["retained_probability"] else 0.0,
            ],
            metrics[selected],
        )

        rows.append({
            "record": record,
            "prospective_selected_channel": selected,
            "channel0": {
                "quality": qualities[0],
                "metrics": metrics[0].to_dict(),
            },
            "channel1": {
                "quality": qualities[1],
                "metrics": metrics[1].to_dict(),
            },
            "channel1_minus_channel0_f1": float(metrics[1].f1 - metrics[0].f1),
            "better_channel_posthoc": (
                1 if metrics[1].f1 > metrics[0].f1 else 0
            ),
            "f1_relation": _f1_relation(metrics[0].f1, metrics[1].f1),
        })
        print(f"[{i:02d}/84] {record}: audited", flush=True)

    report = {
        "schema": "electrotrace.ltafdb_label_free_feature_audit/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "posthoc_exposed_dataset_development_analysis",
        "git_head": git_head(),
        "guardrail": (
            "LTAFDB is exposed development data. Feature/outcome associations in this "
            "report may inform a future selector but cannot validate selector v1 or v2."
        ),
        "inputs": {
            "first_run_report": str(args.first_run_report),
            "first_run_sha256": _prospective.sha256_file(args.first_run_report),
            "model": str(args.model),
            "model_sha256": _prospective.sha256_file(args.model),
            "polarity_derivation": str(args.polarity_threshold_report),
            "polarity_derivation_sha256": _prospective.sha256_file(
                args.polarity_threshold_report
            ),
        },
        "reproduction": {
            "all_84_archived_selected_results_reproduced": True,
            "feature_computation_is_label_free": True,
            "labels_used_only_for_posthoc_channel_outcome_comparison": True,
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
