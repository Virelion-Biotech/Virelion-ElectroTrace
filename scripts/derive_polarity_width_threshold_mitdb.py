#!/usr/bin/env python3
"""Derive the adaptive-polarity width-override confidence cutoff on MIT-BIH development data only.

This script exists to close the evaluation-integrity gap tracked in Issue #14.
It never reads the locked 12-record MIT-BIH test split and never uses INCART.
The selected cutoff must be frozen and recorded in the output before running the
locked evaluator.

The objective is mean record-level Stage-1 F1 on the seven MIT-BIH calibration
records, using only the reference annotations from those development records.
No final held-out metric is computed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, match_peaks
from electrotrace.validation_detectors import (
    DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
    _candidate_set,
    select_signal_polarity,
)
from electrotrace.wfdb_records import load_annotated_record

MITDB_CALIBRATION_RECORDS = ["115", "202", "208", "220", "221", "222", "233"]
LOCKED_HELDOUT_RECORDS = {
    "105", "118", "122", "201", "207", "209", "214", "219", "230", "231", "232", "234"
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def score_cutoff(mitdb_dir: Path, cutoff: float, tolerance_ms: float) -> dict:
    rows = []
    for name in MITDB_CALIBRATION_RECORDS:
        if name in LOCKED_HELDOUT_RECORDS:
            raise RuntimeError(f"refusing to use locked held-out record {name}")
        annotated = load_annotated_record(
            mitdb_dir / name,
            beat_symbols=DEFAULT_BEAT_SYMBOLS,
            tolerance_ms=tolerance_ms,
            policy="error",
        )
        decision = select_signal_polarity(
            annotated.signal,
            annotated.fs_hz,
            width_override_confidence=cutoff,
        )
        z = annotated.signal - np.median(annotated.signal)
        from electrotrace.scale_estimation import estimate_stage1_scale

        scale = estimate_stage1_scale(z, annotated.fs_hz, method="windowed_std")
        candidate_signal = z if decision.polarity != "negative" else -z
        peaks, _ = _candidate_set(candidate_signal, annotated.fs_hz, scale)
        metrics = match_peaks(
            peaks, annotated.reference, annotated.fs_hz, tolerance_ms=tolerance_ms
        )
        rows.append(
            {
                "record": name,
                "polarity": decision.polarity,
                "confidence": decision.confidence,
                "f1": metrics.f1,
                "sensitivity": metrics.sensitivity,
                "positive_predictive_value": metrics.positive_predictive_value,
                "candidate_count": int(len(peaks)),
            }
        )
    return {
        "cutoff": float(cutoff),
        "mean_f1": float(np.mean([r["f1"] for r in rows])),
        "mean_sensitivity": float(np.mean([r["sensitivity"] for r in rows])),
        "mean_ppv": float(np.mean([r["positive_predictive_value"] for r in rows])),
        "records": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mitdb-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tolerance-ms", type=float, default=75.0)
    p.add_argument(
        "--grid",
        type=float,
        nargs="*",
        default=np.round(np.arange(0.05, 0.61, 0.01), 2).tolist(),
    )
    args = p.parse_args()

    grid = sorted({float(x) for x in args.grid if 0.0 <= float(x) <= 1.0})
    if not grid:
        raise SystemExit("empty cutoff grid")
    results = [score_cutoff(args.mitdb_dir, cutoff, args.tolerance_ms) for cutoff in grid]
    selected = max(results, key=lambda r: (r["mean_f1"], r["mean_sensitivity"], -r["cutoff"]))

    report = {
        "schema": "electrotrace.mitdb_polarity_width_threshold_derivation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": "development_only",
        "selected_cutoff": selected["cutoff"],
        "selection_rule": "maximize mean record-level Stage-1 F1 on MIT-BIH calibration records; sensitivity then lower cutoff as deterministic tie-breaks",
        "mitdb_calibration_records": MITDB_CALIBRATION_RECORDS,
        "locked_heldout_records_excluded": sorted(LOCKED_HELDOUT_RECORDS),
        "incart_used": False,
        "tolerance_ms": args.tolerance_ms,
        "default_before_derivation": DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
        "grid": grid,
        "selected_result": selected,
        "all_results": results,
        "freeze_instruction": "Use selected_cutoff as --width-override-confidence in the locked MIT-BIH evaluator; do not retune after inspecting held-out results.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_cutoff": selected["cutoff"],
        "mean_f1": selected["mean_f1"],
        "mean_sensitivity": selected["mean_sensitivity"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
