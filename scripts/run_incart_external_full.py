#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import replace

import numpy as np
import wfdb

from electrotrace import __version__
from electrotrace.candidate_suppressor import (
    CandidateSuppressor,
    _candidate_features,
    label_candidates,
)
from electrotrace.threshold_selection import select_threshold_for_f1
from electrotrace.validation import (
    DEFAULT_BEAT_SYMBOLS,
    DetectionMetrics,
    RecordValidation,
    summarize_records,
    validate_record,
)
from electrotrace.validation_detectors import detect_r_peaks_two_stage


MITDB_TRAIN_RECORDS = [
    "100", "101", "102", "103", "104", "106", "107", "108",
    "109", "111", "112", "113", "114", "115", "116", "117",
    "119", "121", "123", "124", "200", "202", "203", "205",
    "208", "210", "212", "213", "215", "217", "220", "221",
    "222", "223", "228", "233",
]

MITDB_MODEL_FIT_RECORDS = [
    "100", "101", "102", "103", "104", "106", "107", "108",
    "109", "111", "112", "113", "114", "116", "117", "119",
    "121", "123", "124", "200", "203", "205", "210", "212",
    "213", "215", "217", "223", "228",
]

MITDB_CALIBRATION_RECORDS = [
    "115", "202", "208", "220", "221", "222", "233",
]

REFERENCE_BEAT_SYMBOLS = sorted(DEFAULT_BEAT_SYMBOLS)


def provenance():
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        git_head = "unknown"

    versions = {}

    for module_name in ("numpy", "scipy", "sklearn", "wfdb", "pandas"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "missing"

    return {
        "git_head": git_head,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "electrotrace_version": __version__,
        "package_versions": versions,
    }


def load_record(record, data_dir):
    base = str(Path(data_dir) / record)

    rec = wfdb.rdrecord(
        base,
        channels=[0],
        physical=False,
    )

    signal = np.asarray(
        rec.p_signal[:, 0]
        if rec.p_signal is not None
        else rec.d_signal[:, 0],
        dtype=float,
    )

    ann = wfdb.rdann(base, "atr")

    refs = np.asarray(
        [
            int(sample)
            for sample, symbol in zip(ann.sample, ann.symbol)
            if symbol in REFERENCE_BEAT_SYMBOLS
        ],
        dtype=int,
    )

    if refs.size:
        if np.any(refs < 0):
            raise ValueError(
                f"{record}: negative annotation sample index encountered."
            )

    return base, rec, signal, refs


def candidate_stream(signal, fs_hz, polarity="adaptive", scale_method=None):
    from electrotrace.validation_detectors import (
        DEFAULT_SCALE_METHOD,
        _candidate_set,
        detect_r_peaks,
        estimate_stage1_scale,
        select_signal_polarity,
    )

    scale_method = scale_method or DEFAULT_SCALE_METHOD
    chosen = polarity

    if chosen == "adaptive":
        chosen = select_signal_polarity(signal, fs_hz, scale_method=scale_method).polarity

    primary = detect_r_peaks(
        signal,
        fs_hz,
        polarity=chosen,
        scale_method=scale_method,
    )

    if len(primary) == 0:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
            chosen,
        )

    z = signal - np.median(signal)
    scale = estimate_stage1_scale(z, fs_hz, method=scale_method)

    candidate_signal = (
        z if chosen != "negative" else -z
    )

    primary, prominences = _candidate_set(
        candidate_signal,
        fs_hz,
        scale,
    )

    return primary, prominences, chosen


def train_mitdb_model(data_dir, seed=42, scale_method=None):
    all_features = []
    all_labels = []
    all_groups = []
    feature_names = None

    for record in MITDB_MODEL_FIT_RECORDS:
        _, rec, signal, references = load_record(
            record,
            data_dir,
        )

        candidates, prominences, polarity = candidate_stream(
            signal,
            float(rec.fs),
            polarity="adaptive",
            scale_method=scale_method,
        )

        features, names = _candidate_features(
            signal,
            float(rec.fs),
            candidates,
            prominences,
        )

        labels = label_candidates(
            candidates,
            references,
            float(rec.fs),
        )

        all_features.append(features)
        all_labels.append(labels)
        all_groups.append(
            np.full(
                len(labels),
                record,
                dtype=object,
            )
        )

        feature_names = names

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    model = CandidateSuppressor().fit(
        X,
        y,
        target_recall=0.995,
        random_seed=seed,
        calibration_fraction=0,
        n_estimators=150,
    )

    calibration_features = []
    calibration_labels = []

    for record in MITDB_CALIBRATION_RECORDS:
        _, rec, signal, references = load_record(
            record,
            data_dir,
        )

        candidates, prominences, _ = candidate_stream(
            signal,
            float(rec.fs),
            polarity="adaptive",
            scale_method=scale_method,
        )

        features, _ = _candidate_features(
            signal,
            float(rec.fs),
            candidates,
            prominences,
        )

        labels = label_candidates(
            candidates,
            references,
            float(rec.fs),
        )

        calibration_features.append(features)
        calibration_labels.append(labels)

    X_cal = np.vstack(calibration_features)
    y_cal = np.concatenate(calibration_labels)

    calibration_probabilities = model.predict_proba(X_cal)

    threshold = select_threshold_for_f1(
        y_cal,
        calibration_probabilities,
        min_recall=0.97,
    )

    model.metadata = replace(
        model.metadata,
        threshold=float(threshold),
        calibration_fraction=float(len(y_cal) / len(y)),
        calibration_candidates=int(len(y_cal)),
        calibration_method="locked_MITBIH_calibration_records",
    )

    model.feature_names = feature_names

    return model


def validate_model_on_incart(
    model,
    incart_dir,
    tolerance_ms=75.0,
    scale_method=None,
):
    raw_records = wfdb.get_record_list("incartdb")

    record_names = [
        Path(r).name
        for r in raw_records
    ]

    results = []
    skipped = []

    for idx, record in enumerate(record_names, start=1):
        base = str(Path(incart_dir) / record)

        try:
            _, rec, signal, references = load_record(
                record,
                incart_dir,
            )

            if references.size and np.any(references < 0):
                skipped.append({
                    "record": record,
                    "reason": "negative annotation sample index",
                })
                continue

            def detector(test_signal, fs_hz):
                retained, _ = detect_r_peaks_two_stage(
                    test_signal,
                    fs_hz,
                    model,
                    polarity="adaptive",
                    recovery=False,
                    scale_method=scale_method,
                )

                return retained

            result = validate_record(
                base,
                detector,
                channel=0,
                annotation_extension="atr",
                beat_symbols=REFERENCE_BEAT_SYMBOLS,
                tolerance_ms=tolerance_ms,
            )

            payload = result.to_dict()

            candidates, _, chosen_polarity = candidate_stream(
                signal,
                float(rec.fs),
                polarity="adaptive",
                scale_method=scale_method,
            )

            retained = detector(
                signal,
                float(rec.fs),
            )

            payload["stage1_candidates"] = int(
                len(candidates)
            )

            payload["stage2_retained"] = int(
                len(retained)
            )

            payload["suppression_rate"] = (
                float(
                    1.0
                    - len(retained) / len(candidates)
                )
                if len(candidates)
                else 0.0
            )

            payload["selected_polarity"] = chosen_polarity

            results.append(payload)

            print(
                f"[{idx}/{len(record_names)}] "
                f"{record}: "
                f"F1={payload['f1']:.4f}, "
                f"Sens={payload['sensitivity']:.4f}, "
                f"PPV={payload['positive_predictive_value']:.4f}"
            )

        except Exception as exc:
            skipped.append({
                "record": record,
                "reason": f"{type(exc).__name__}: {exc}",
            })

            print(
                f"[{idx}/{len(record_names)}] "
                f"{record}: SKIPPED: {exc}"
            )

    return results, skipped


def payload_to_validation(payload):
    metrics = DetectionMetrics(
        reference_count=int(payload["reference_count"]),
        detected_count=int(payload["detected_count"]),
        true_positive=int(payload["true_positive"]),
        false_positive=int(payload["false_positive"]),
        false_negative=int(payload["false_negative"]),
        sensitivity=float(payload["sensitivity"]),
        positive_predictive_value=float(
            payload["positive_predictive_value"]
        ),
        f1=float(payload["f1"]),
        mean_timing_error_ms=payload.get(
            "mean_timing_error_ms"
        ),
        median_timing_error_ms=payload.get(
            "median_timing_error_ms"
        ),
        timing_sd_ms=payload.get(
            "timing_sd_ms"
        ),
        median_absolute_timing_error_ms=payload.get(
            "median_absolute_timing_error_ms"
        ),
        mean_absolute_timing_error_ms=payload.get(
            "mean_absolute_timing_error_ms"
        ),
        p95_absolute_timing_error_ms=payload.get(
            "p95_absolute_timing_error_ms"
        ),
        max_absolute_timing_error_ms=payload.get(
            "max_absolute_timing_error_ms"
        ),
    )

    return RecordValidation(
        record=payload["record"],
        fs_hz=float(payload["fs_hz"]),
        metrics=metrics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale-method",
        default=None,
        choices=["std", "mad", "windowed_mad", "windowed_std", "adaptive"],
        help=(
            "Stage-1 amplitude-scale estimator. Default (None) uses "
            "electrotrace.validation_detectors.DEFAULT_SCALE_METHOD "
            "('windowed_mad', the Priority-1 fix). Pass 'std' to reproduce "
            "the pre-fix behavior for an A/B comparison run."
        ),
    )
    args = parser.parse_args()

    from electrotrace.validation_detectors import DEFAULT_SCALE_METHOD
    scale_method = args.scale_method or DEFAULT_SCALE_METHOD

    data_dir = Path(".cache/physionet/mitdb")
    incart_dir = Path(".cache/physionet/incartdb")

    output_dir = Path("validation_reports")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not incart_dir.exists():
        raise SystemExit(
            f"Missing INCART directory: {incart_dir}"
        )

    print(f"Training locked MIT-BIH model... (scale_method={scale_method})")
    model = train_mitdb_model(
        data_dir,
        seed=42,
        scale_method=scale_method,
    )

    model_path = (
        output_dir /
        f"incart_mitbih_model_{scale_method}_2026-09-09.pkl"
    )

    model.save(model_path)

    print("Model saved:", model_path)

    results, skipped = validate_model_on_incart(
        model,
        incart_dir,
        tolerance_ms=75.0,
        scale_method=scale_method,
    )

    validation_objects = [
        payload_to_validation(x)
        for x in results
    ]

    summary = summarize_records(
        validation_objects
    )

    report = {
        "schema": "electrotrace.external_incart_validation/v2",
        "software_version": __version__,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": {
            "name": "PhysioNet INCART",
            "database": "incartdb",
            "expected_records": 75,
            "records_returned": len(
                wfdb.get_record_list("incartdb")
            ),
        },
        "provenance": provenance(),
        "protocol": {
            "tolerance_ms": 75,
            "channel": 0,
            "annotation_extension": "atr",
            "beat_symbols": REFERENCE_BEAT_SYMBOLS,
            "model_training_database": "MIT-BIH",
            "model_training_records": MITDB_TRAIN_RECORDS,
            "model_fit_records": MITDB_MODEL_FIT_RECORDS,
            "model_calibration_records": MITDB_CALIBRATION_RECORDS,
            "model_seed": 42,
            "n_estimators": 150,
            "polarity": "adaptive",
            "recovery": False,
            "stage1_scale_method": scale_method,
            "evaluation": "external_record_level",
            "incart_annotations_used_for_model_selection": False,
        },
        "model": {
            "threshold": float(
                model.metadata.threshold
            ),
            "metadata": model.metadata.to_dict(),
            "path": str(model_path),
        },
        "record_results": results,
        "skipped_records": skipped,
        "summary": summary,
    }

    output = (
        output_dir /
        f"incart_two_stage_external_full_{scale_method}_2026-09-09.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))

    print(
        "\nReport written to:",
        output,
    )

    print(
        "Skipped records:",
        len(skipped),
    )

    if len(results) == 0:
        raise SystemExit(
            "No usable INCART records. "
            "Do not interpret this run."
        )


if __name__ == "__main__":
    main()
