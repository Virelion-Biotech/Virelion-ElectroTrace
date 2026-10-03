#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_base_validator():
    path = (
        Path(__file__).resolve().parent /
        "validate_qtdb.py"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Required helper missing: {path}. "
            "Ensure scripts/validate_qtdb.py is in the repository."
        )

    spec = importlib.util.spec_from_file_location(
        "validate_qtdb",
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules["validate_qtdb"] = module

    spec.loader.exec_module(module)

    return module


def run_one(
    module,
    dataset_dir,
    tolerance_ms,
    channel,
):
    result = {}

    # Reuse the exact existing detector,
    # reference parser and delineator.
    wfdb = module.require_wfdb()

    records = sorted(
        dataset_dir.glob("*.hea")
    )

    if not records:
        raise RuntimeError(
            "No QTDB WFDB records found."
        )

    onset_errors = []
    offset_errors = []

    per_record = []

    for header in records:
        base = header.with_suffix("")

        record = wfdb.rdrecord(
            str(base),
            channels=[channel],
            physical=True,
        )

        signal = record.p_signal[:, 0]

        ref_onset, ref_offset = (
            module.qrs_reference(
                wfdb,
                base,
            )
        )

        detector = (
            module.load_detector(
                "electrotrace.qtdb_detector_adapter:"
                "detect_r_peaks_adaptive"
            )
        )

        detected = detector(
            signal,
            float(record.fs),
        )

        onset_err, offset_err, pairs = (
            module.boundary_errors(
                signal,
                float(record.fs),
                detected,
                ref_onset,
                ref_offset,
                tolerance_ms,
            )
        )

        onset_errors.extend(
            onset_err
        )
        offset_errors.extend(
            offset_err
        )

        per_record.append(
            {
                "record": base.name,
                "fs_hz": float(record.fs),
                "reference_qrs": len(ref_onset),
                "detected_qrs": len(detected),
                "matched_qrs": len(pairs),
                "onset": module.summarize(
                    onset_err
                ),
                "offset": module.summarize(
                    offset_err
                ),
            }
        )

    return {
        "tolerance_ms": tolerance_ms,
        "records": len(per_record),
        "summary": {
            "reference_qrs": sum(
                x["reference_qrs"]
                for x in per_record
            ),
            "detected_qrs": sum(
                x["detected_qrs"]
                for x in per_record
            ),
            "matched_qrs": sum(
                x["matched_qrs"]
                for x in per_record
            ),
            "onset": module.summarize(
                onset_errors
            ),
            "offset": module.summarize(
                offset_errors
            ),
        },
        "records_detail": per_record,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_dir",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--tolerances",
        default="20,40,60,80,100",
    )

    args = parser.parse_args()

    module = load_base_validator()

    tolerances = [
        float(x.strip())
        for x in args.tolerances.split(",")
        if x.strip()
    ]

    results = []

    for tolerance in tolerances:
        print(
            f"\n===== tolerance "
            f"{tolerance:g} ms =====",
            flush=True,
        )

        result = run_one(
            module,
            args.dataset_dir,
            tolerance,
            args.channel,
        )

        results.append(result)

        print(
            json.dumps(
                result["summary"],
                indent=2,
            ),
            flush=True,
        )

    final = {
        "schema": (
            "electrotrace.qtdb_delineation_"
            "tolerance_curves/v1"
        ),
        "created_at_utc": (
            __import__("datetime")
            .datetime.now(
                __import__("datetime")
                .timezone.utc
            )
            .isoformat()
        ),
        "dataset": getattr(module, "DATASET_VERSION", "qtdb"),
        "detector": (
            "electrotrace.qtdb_detector_adapter:"
            "detect_r_peaks_adaptive"
        ),
        "delineator_version": getattr(
            module, "DELINEATOR_VERSION", "unknown"
        ),
        "channel": args.channel,
        "tolerances_ms": tolerances,
        "results": results,
        "protocol_note": (
            "The detector and QRS delineator are "
            "unchanged across tolerance levels; "
            "only the event-matching tolerance "
            "is varied."
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            final,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\nWritten:",
        args.output,
    )


if __name__ == "__main__":
    main()
