#!/usr/bin/env python3
"""Train and evaluate ElectroTrace's two-stage MIT-BIH R-peak detector.

The first stage is the existing high-recall candidate detector. The second
stage is trained only on training records and then evaluated on held-out
records, preventing record-level leakage.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from electrotrace import __version__
from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features, label_candidates
from electrotrace.validation import summarize_records, validate_record
from electrotrace.validation_detectors import detect_r_peaks


def _record_detector(signal: np.ndarray, fs_hz: float) -> np.ndarray:
    return detect_r_peaks(signal, fs_hz)


def _train_records(records: list[str], data_dir: Path):
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for record in records:
        base = str(data_dir / record)
        rec = wfdb.rdrecord(base, channels=[0], physical=False)
        signal = np.asarray(rec.p_signal[:, 0] if rec.p_signal is not None else rec.d_signal[:, 0], dtype=float)
        candidates, properties = __import__("scipy").signal.find_peaks(
            signal - np.median(signal),
            distance=max(1, int(round(float(rec.fs) * 0.25))),
            prominence=float(np.std(signal)) * 0.5,
        )
        ann = wfdb.rdann(base, "atr")
        symbols = np.asarray(ann.symbol)
        allowed = {"/", "A", "E", "F", "J", "L", "N", "Q", "R", "S", "V", "a", "e", "f", "j"}
        refs = np.asarray([s for s, symbol in zip(ann.sample, symbols) if symbol in allowed], dtype=int)
        X, names = _candidate_features(signal, float(rec.fs), candidates, properties["prominences"])
        y = label_candidates(candidates, refs, float(rec.fs))
        features.append(X)
        labels.append(y)
        feature_names = names
    model = CandidateSuppressor().fit(np.vstack(features), np.concatenate(labels), target_recall=0.995)
    model.feature_names = feature_names
    return model


def _evaluate(model: CandidateSuppressor, records: list[str], data_dir: Path) -> list:
    results = []
    for record in records:
        base = str(data_dir / record)
        rec = wfdb.rdrecord(base, channels=[0], physical=False)
        signal = np.asarray(rec.p_signal[:, 0] if rec.p_signal is not None else rec.d_signal[:, 0], dtype=float)
        candidates, properties = __import__("scipy").signal.find_peaks(
            signal - np.median(signal),
            distance=max(1, int(round(float(rec.fs) * 0.25))),
            prominence=float(np.std(signal)) * 0.5,
        )
        features, _ = _candidate_features(signal, float(rec.fs), candidates, properties["prominences"])
        retained, probabilities = model.filter_candidates(candidates, features)
        # Reuse ElectroTrace's established one-to-one validation metrics.
        def detector(_signal, _fs, retained=retained):
            return retained
        result = validate_record(base, detector, channel=0, annotation_extension="atr", tolerance_ms=75)
        result = result.to_dict()
        result["stage1_detected"] = int(len(candidates))
        result["stage2_retained"] = int(len(retained))
        result["suppression_rate"] = float(1.0 - len(retained) / len(candidates)) if len(candidates) else 0.0
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing the MIT-BIH WFDB files")
    parser.add_argument("--output", default="validation_reports/mitdb_two_stage_validation.json")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    records = list(wfdb.get_record_list("mitdb"))
    rng = np.random.default_rng(args.seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * args.test_fraction)))
    test_records = sorted(shuffled[:n_test])
    train_records = sorted(shuffled[n_test:])

    model = _train_records(train_records, data_dir)
    record_results = _evaluate(model, test_records, data_dir)

    # Aggregate from the record-level TP/FP/FN counts; this remains the primary
    # scientific endpoint for the held-out detector comparison.
    summary = summarize_records([
        type("ValidationResult", (), {
            "metrics": type("Metrics", (), {
                "reference_count": r["reference_count"],
                "detected_count": r["detected_count"],
                "true_positive": r["true_positive"],
                "false_positive": r["false_positive"],
                "false_negative": r["false_negative"],
                "median_timing_error_ms": r["median_timing_error_ms"],
            })()
        })()
        for r in record_results
    ])
    report = {
        "schema": "electrotrace.two_stage_validation/v1",
        "software_version": __version__,
        "train_records": train_records,
        "test_records": test_records,
        "target_recall": model.metadata.target_recall,
        "threshold": model.metadata.threshold,
        "model_metadata": model.metadata.to_dict(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_results": record_results,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model.save(output.with_suffix(".pkl"))
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to {output}")
    print(f"Model written to {output.with_suffix('.pkl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
