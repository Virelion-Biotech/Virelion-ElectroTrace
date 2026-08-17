#!/usr/bin/env python3
"""Train and evaluate ElectroTrace's two-stage MIT-BIH R-peak detector."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb
from sklearn.model_selection import StratifiedGroupKFold

from electrotrace import __version__
from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features, label_candidates, select_threshold_for_recall
from electrotrace.validation import DetectionMetrics, RecordValidation, summarize_records, validate_record
from electrotrace.validation_detectors import detect_r_peaks_two_stage

ALLOWED_BEAT_SYMBOLS = {"/", "A", "E", "F", "J", "L", "N", "Q", "R", "S", "V", "a", "e", "f", "j"}


def _load_record(record: str, data_dir: Path):
    base = str(data_dir / record)
    rec = wfdb.rdrecord(base, channels=[0], physical=False)
    signal = np.asarray(rec.p_signal[:, 0] if rec.p_signal is not None else rec.d_signal[:, 0], dtype=float)
    ann = wfdb.rdann(base, "atr")
    refs = np.asarray([s for s, symbol in zip(ann.sample, ann.symbol) if symbol in ALLOWED_BEAT_SYMBOLS], dtype=int)
    return base, rec, signal, refs


def _candidate_stream(signal: np.ndarray, fs_hz: float, polarity: str, recovery: bool):
    from electrotrace.validation_detectors import _candidate_set, detect_r_peaks, recover_stage1_candidates, select_signal_polarity
    chosen = polarity
    if chosen == "adaptive":
        chosen = select_signal_polarity(signal, fs_hz).polarity
    primary = detect_r_peaks(signal, fs_hz, polarity=chosen)
    if len(primary) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float), chosen
    z = signal - np.median(signal)
    scale = float(np.std(z))
    candidate_signal = z if chosen != "negative" else -z
    primary, prominences = _candidate_set(candidate_signal, fs_hz, scale)
    if not recovery:
        return primary, prominences, chosen
    extra, extra_prom = recover_stage1_candidates(signal, fs_hz, primary, polarity=chosen)
    if len(extra) == 0:
        return primary, prominences, chosen
    all_peaks = np.sort(np.concatenate([primary, extra]))
    prom_map = {int(i): float(p) for i, p in zip(primary, prominences)}
    prom_map.update({int(i): float(p) for i, p in zip(extra, extra_prom)})
    all_prom = np.asarray([prom_map[int(i)] for i in all_peaks], dtype=float)
    return all_peaks, all_prom, chosen


def _fit_group_calibrated(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, target_recall: float, seed: int) -> tuple[CandidateSuppressor, list[str]]:
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("record-level calibration requires at least three training records")
    splitter = StratifiedGroupKFold(n_splits=min(5, len(unique_groups)), shuffle=True, random_state=seed)
    fit_idx, calibration_idx = next(splitter.split(features, labels, groups))
    calibration_groups = sorted({str(value) for value in groups[calibration_idx]})
    if set(groups[fit_idx]).intersection(set(groups[calibration_idx])):
        raise RuntimeError("calibration records overlap model-fitting records")
    if len(np.unique(labels[fit_idx])) < 2 or len(np.unique(labels[calibration_idx])) < 2:
        raise ValueError("record-level calibration split must contain both positive and negative candidates")

    calibration_model = CandidateSuppressor().fit(features[fit_idx], labels[fit_idx], target_recall=target_recall, random_seed=seed, calibration_fraction=0)
    calibration_probabilities = calibration_model.predict_proba(features[calibration_idx])
    threshold = select_threshold_for_recall(labels[calibration_idx], calibration_probabilities, target_recall=target_recall)

    final_model = CandidateSuppressor().fit(features, labels, target_recall=target_recall, random_seed=seed, calibration_fraction=0)
    final_model.metadata = replace(
        final_model.metadata,
        threshold=float(threshold),
        calibration_fraction=float(len(calibration_idx) / len(labels)),
        calibration_candidates=int(len(calibration_idx)),
        calibration_method="held_out_record_group",
    )
    return final_model, calibration_groups


def _train_records(records: list[str], data_dir: Path, polarity: str, recovery: bool, seed: int) -> tuple[CandidateSuppressor, list[str]]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for record in records:
        _, rec, signal, refs = _load_record(record, data_dir)
        candidates, prominences, _ = _candidate_stream(signal, float(rec.fs), polarity, recovery)
        X, names = _candidate_features(signal, float(rec.fs), candidates, prominences)
        y = label_candidates(candidates, refs, float(rec.fs))
        features.append(X)
        labels.append(y)
        groups.append(np.full(len(y), record, dtype=object))
        feature_names = names
    matrix = np.vstack(features)
    target = np.concatenate(labels)
    group_ids = np.concatenate(groups)
    model, calibration_records = _fit_group_calibrated(matrix, target, group_ids, target_recall=0.995, seed=seed)
    model.feature_names = feature_names
    return model, calibration_records


def _evaluate(model: CandidateSuppressor, records: list[str], data_dir: Path, polarity: str, recovery: bool) -> list[dict]:
    results: list[dict] = []
    for record in records:
        base, rec, signal, _ = _load_record(record, data_dir)

        def detector(test_signal: np.ndarray, fs_hz: float) -> np.ndarray:
            retained, _ = detect_r_peaks_two_stage(test_signal, fs_hz, model, polarity=polarity, recovery=recovery)
            return retained

        result = validate_record(base, detector, channel=0, annotation_extension="atr", tolerance_ms=75)
        payload = result.to_dict()
        candidates, _, _ = _candidate_stream(signal, float(rec.fs), polarity, recovery)
        retained = detector(signal, float(rec.fs))
        payload["stage1_detected"] = int(len(candidates))
        payload["stage2_retained"] = int(len(retained))
        payload["suppression_rate"] = float(1.0 - len(retained) / len(candidates)) if len(candidates) else 0.0
        results.append(payload)
    return results


def _summary_from_payloads(payloads: list[dict]) -> dict:
    results = [RecordValidation(record=p["record"], fs_hz=float(p["fs_hz"]), metrics=DetectionMetrics(reference_count=int(p["reference_count"]), detected_count=int(p["detected_count"]), true_positive=int(p["true_positive"]), false_positive=int(p["false_positive"]), false_negative=int(p["false_negative"]), sensitivity=float(p["sensitivity"]), positive_predictive_value=float(p["positive_predictive_value"]), f1=float(p["f1"]), median_timing_error_ms=p["median_timing_error_ms"], p95_timing_error_ms=p["p95_timing_error_ms"])) for p in payloads]
    return summarize_records(results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="validation_reports/mitdb_two_stage_validation.json")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--polarity", choices=["positive", "negative", "adaptive"], default="positive")
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        raise SystemExit("--test-fraction must be between 0 and 1")

    data_dir = Path(args.data_dir)
    records = list(wfdb.get_record_list("mitdb"))
    rng = np.random.default_rng(args.seed)
    shuffled = records.copy(); rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * args.test_fraction)))
    test_records = sorted(shuffled[:n_test]); train_records = sorted(shuffled[n_test:])

    model, calibration_records = _train_records(train_records, data_dir, args.polarity, args.recovery, args.seed)
    record_results = _evaluate(model, test_records, data_dir, args.polarity, args.recovery)
    report = {
        "schema": "electrotrace.two_stage_validation/v4",
        "software_version": __version__,
        "train_records": train_records,
        "calibration_records": calibration_records,
        "test_records": test_records,
        "polarity": args.polarity,
        "recovery": bool(args.recovery),
        "target_recall": model.metadata.target_recall,
        "threshold": model.metadata.threshold,
        "model_metadata": model.metadata.to_dict(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_results": record_results,
        "summary": _summary_from_payloads(record_results),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model.save(output.with_suffix(".pkl"))
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
