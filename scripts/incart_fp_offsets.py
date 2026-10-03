#!/usr/bin/env python3
"""Where do the extra detections come from?  False-positive forensics on INCART.

Runs the *frozen* two-stage model (no retraining, no threshold selection) over
INCART and, for every false positive at the model's operating threshold, records
where it sits relative to the neighbouring reference beats, how tall it is
compared with the nearest true beat, and what Stage 2 thought of it.  It also
reports how well the Stage-2 probabilities rank true against false candidates
(AUC) and the best global threshold each record *could* have had (oracle,
diagnostic only).

Evidence status: INCART is development data for this model generation (scale
variants and thresholds were examined against it).  Nothing here is a test
result.  Do not run this on a database you have reserved as a held-out test.

Usage (Colab, INCART cached locally):
  python -u scripts/incart_fp_offsets.py \\
      --incart-dir .cache/physionet/incartdb \\
      --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops

Outputs (in --output-dir): incart_fp_forensics.json, incart_fp_detail.csv and,
if matplotlib is available, incart_fp_forensics.png.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from electrotrace import __version__
from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.fp_analysis import (
    CATEGORIES,
    DEFAULT_BANDS_MS,
    analyze_record,
    match_peaks_detailed,
    pool_false_positives,
)
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS
from electrotrace.validation_detectors import detect_r_peaks_two_stage
from electrotrace.wfdb_records import (
    POLICIES,
    RecordExcluded,
    load_annotated_record,
    summarize_audits,
)

NUMERIC_COLUMNS = ("dt_prev_ms", "dt_next_ms", "rr_ms", "phase", "nearest_offset_ms")

EVIDENCE_NOTE = (
    "Development diagnostic. INCART informed the design of this model generation, so nothing "
    "in this report is a held-out test result. Oracle thresholds use reference labels and are "
    "an upper bound for diagnosis, not a deployable setting."
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


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
    for module_name in ("numpy", "scipy", "sklearn", "skops", "wfdb", "pandas", "matplotlib"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "missing"
    return versions


def analyze_signal_record(
    model,
    signal: np.ndarray,
    fs_hz: float,
    reference: np.ndarray,
    *,
    scale_method: str = "windowed_std",
    threshold: float | None = None,
    tolerance_ms: float = 75.0,
) -> dict:
    """Score every Stage-1 candidate once, then analyse the false positives.

    ``threshold=0.0`` retains every Stage-1 candidate, so the returned candidates
    are exactly the Stage-1 set (this is also how the Phase-3 ablation scores).
    """
    operating = float(model.metadata.threshold if threshold is None else threshold)
    candidates, probabilities = detect_r_peaks_two_stage(
        signal,
        fs_hz,
        model,
        polarity="adaptive",
        recovery=False,
        scale_method=scale_method,
        threshold=0.0,
    )
    candidates = np.asarray(candidates, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    result = analyze_record(
        signal,
        fs_hz,
        candidates,
        probabilities,
        reference,
        threshold=operating,
        tolerance_ms=tolerance_ms,
    )
    truth = match_peaks_detailed(candidates, reference, fs_hz, tolerance_ms).detected_matched
    result["stage1_candidates"] = int(candidates.size)
    result["candidate_probability_true"] = probabilities[truth]
    result["candidate_probability_false"] = probabilities[~truth]
    return result


def record_row(name: str, fs_hz: float, result: dict) -> dict:
    m = result["metrics"]
    cand = result["candidates"]
    return {
        "record": name,
        "fs_hz": fs_hz,
        "reference_count": m["reference_count"],
        "stage1_candidates": result["stage1_candidates"],
        "detected_count": m["detected_count"],
        "sensitivity": m["sensitivity"],
        "ppv": m["positive_predictive_value"],
        "f1": m["f1"],
        "false_positive": m["false_positive"],
        "detected_over_reference": result["detected_over_reference"],
        "false_positive_categories": result["false_positive_categories"],
        "stage2_auc": cand["auc"],
        "oracle_best_f1": cand["oracle_best_f1"],
        "oracle_best_threshold": cand["oracle_best_threshold"],
        "prob_true_candidates": cand["prob_true"],
        "prob_false_candidates": cand["prob_false"],
    }


def group_summary(rows: list[dict], results: dict[str, dict]) -> dict:
    if not rows:
        return {"n_records": 0}
    aucs = [r["stage2_auc"] for r in rows if r["stage2_auc"] is not None]
    oracle = [r["oracle_best_f1"] for r in rows if r["oracle_best_f1"] is not None]
    return {
        "n_records": len(rows),
        "records": [r["record"] for r in rows],
        "mean_f1_at_model_threshold": float(np.mean([r["f1"] for r in rows])),
        "mean_sensitivity": float(np.mean([r["sensitivity"] for r in rows])),
        "mean_ppv": float(np.mean([r["ppv"] for r in rows])),
        "median_stage2_auc": float(np.median(aucs)) if aucs else None,
        "mean_oracle_best_f1": float(np.mean(oracle)) if oracle else None,
        "oracle_is_diagnostic_only": True,
        "pooled_false_positives": pool_false_positives(
            [results[r["record"]]["false_positive_table"] for r in rows]
        ),
    }


def write_detail_csv(
    path: Path,
    results: dict[str, dict],
    fs_by_record: dict[str, float],
    groups: dict,
) -> None:
    columns = [
        "record",
        "group",
        "sample",
        "time_s",
        "category",
        "dt_prev_ms",
        "dt_next_ms",
        "rr_ms",
        "phase",
        "nearest_offset_ms",
        "amp_ratio",
        "same_polarity",
        "stage2_probability",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for name, result in results.items():
            t = result["false_positive_table"]
            fs = fs_by_record[name]
            for k in range(len(t["sample"])):
                writer.writerow(
                    [
                        name,
                        groups[name],
                        int(t["sample"][k]),
                        f"{t['sample'][k] / fs:.3f}",
                        t["category"][k],
                        *(f"{t[c][k]:.3f}" for c in NUMERIC_COLUMNS),
                        f"{t['amp_ratio'][k]:.3f}",
                        int(bool(t["same_polarity"][k])),
                        f"{t['probability'][k]:.4f}",
                    ]
                )


def make_figure(path: Path, results: dict[str, dict], groups: dict, threshold: float) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    def stack(key: str, group: str, source: str = "false_positive_table") -> np.ndarray:
        parts = [r[source][key] if source else r[key] for n, r in results.items() if groups[n] == group]
        return np.concatenate(parts) if parts else np.array([])

    def hist_fraction(ax, values, bins, colour, label):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(v, bins=bins, weights=np.full(v.size, 1.0 / v.size), alpha=0.6, color=colour, label=label)

    over, other = "overdetecting", "other"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for group, colour in ((over, "tab:red"), (other, "tab:gray")):
        hist_fraction(axes[0, 0], stack("nearest_offset_ms", group), np.arange(-600, 601, 25), colour, group)
        hist_fraction(axes[0, 1], stack("phase", group), np.linspace(0, 1, 21), colour, group)
    axes[0, 0].set_title("False positives: signed offset to nearest reference beat")
    axes[0, 0].set_xlabel("ms (detection minus reference)")
    axes[0, 0].set_ylabel("fraction of group's false positives")
    axes[0, 0].legend()
    axes[0, 1].set_title("Position within the RR interval (0 = previous beat)")
    axes[0, 1].set_xlabel("phase")
    axes[0, 1].legend()

    fractions = []
    for group in (over, other):
        cats = stack("category", group)
        total = max(len(cats), 1)
        fractions.append([float(np.sum(cats == c)) / total for c in CATEGORIES])
    x = np.arange(len(CATEGORIES))
    axes[1, 0].bar(x - 0.2, fractions[0], width=0.4, color="tab:red", label=over)
    axes[1, 0].bar(x + 0.2, fractions[1], width=0.4, color="tab:gray", label=other)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([c.replace("_", "\n") for c in CATEGORIES], fontsize=7)
    axes[1, 0].set_title("False-positive timing category")
    axes[1, 0].legend()

    true_p = stack("candidate_probability_true", over, source="")
    false_p = stack("candidate_probability_false", over, source="")
    bins = np.linspace(0, 1, 41)
    if true_p.size:
        axes[1, 1].hist(true_p, bins=bins, alpha=0.6, color="tab:green", label="true candidates")
    if false_p.size:
        axes[1, 1].hist(false_p, bins=bins, alpha=0.6, color="tab:red", label="false candidates")
    axes[1, 1].axvline(threshold, color="k", linestyle="--", label=f"threshold {threshold:.3f}")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Stage-2 probability, over-detecting records")
    axes[1, 1].legend()
    fig.suptitle("INCART false-positive forensics (development data, not a test result)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incart-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_reports/experiments/2026-09-incart-fp-forensics"),
    )
    parser.add_argument("--records", nargs="*", default=None, help="Defaults to every local INCART record.")
    parser.add_argument("--scale-method", default="windowed_std")
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Operating threshold to analyse. Defaults to the model's stored threshold.",
    )
    parser.add_argument(
        "--overdetect-ratio",
        type=float,
        default=1.5,
        help="Records with detected/reference above this form the 'overdetecting' group.",
    )
    parser.add_argument("--annotation-policy", choices=POLICIES, default="error")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    model = CandidateSuppressor.load(args.model)
    threshold = float(model.metadata.threshold if args.threshold is None else args.threshold)
    names = args.records or sorted(p.stem for p in args.incart_dir.glob("*.hea"))
    if not names:
        raise SystemExit(f"No records found under {args.incart_dir}")

    rows: list[dict] = []
    results: dict[str, dict] = {}
    fs_by_record: dict[str, float] = {}
    skipped: list[dict] = []
    audits = []

    for index, name in enumerate(names, start=1):
        try:
            annotated = load_annotated_record(
                args.incart_dir / name,
                beat_symbols=DEFAULT_BEAT_SYMBOLS,
                tolerance_ms=args.tolerance_ms,
                policy=args.annotation_policy,
            )
            audits.append(annotated.audit)
            result = analyze_signal_record(
                model,
                annotated.signal,
                annotated.fs_hz,
                annotated.reference,
                scale_method=args.scale_method,
                threshold=threshold,
                tolerance_ms=args.tolerance_ms,
            )
        except RecordExcluded as exc:
            audits.append(exc.audit)
            skipped.append({"record": name, "reason": exc.reason})
            print(f"[{index}/{len(names)}] {name}: EXCLUDED: {exc.reason}", flush=True)
            continue
        except Exception as exc:
            skipped.append({"record": name, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(names)}] {name}: SKIPPED: {exc}", flush=True)
            continue
        row = record_row(name, annotated.fs_hz, result)
        rows.append(row)
        results[name] = result
        fs_by_record[name] = annotated.fs_hz
        top = max(row["false_positive_categories"].items(), key=lambda kv: kv[1])
        print(
            f"[{index}/{len(names)}] {name}: F1={row['f1']:.3f} sens={row['sensitivity']:.3f} "
            f"ppv={row['ppv']:.3f} det/ref={row['detected_over_reference']:.2f} "
            f"AUC={row['stage2_auc'] if row['stage2_auc'] is None else round(row['stage2_auc'], 3)} "
            f"top FP category={top[0]} ({top[1]})",
            flush=True,
        )

    if not rows:
        raise SystemExit("No usable records; nothing to report.")

    groups = {
        r["record"]: "overdetecting" if (r["detected_over_reference"] or 0.0) > args.overdetect_ratio else "other"
        for r in rows
    }
    over_rows = [r for r in rows if groups[r["record"]] == "overdetecting"]
    other_rows = [r for r in rows if groups[r["record"]] == "other"]
    for r in rows:
        r["group"] = groups[r["record"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "incart_fp_detail.csv"
    png_path = args.output_dir / "incart_fp_forensics.png"
    json_path = args.output_dir / "incart_fp_forensics.json"
    write_detail_csv(csv_path, results, fs_by_record, groups)
    plotted = False if args.no_plot else make_figure(png_path, results, groups, threshold)

    input_hashes = {}
    for name in [r["record"] for r in rows]:
        for source in sorted(args.incart_dir.glob(f"{name}.*")):
            if source.is_file():
                input_hashes[f"{name}::{source.name}"] = sha256_file(source)

    report = {
        "schema": "electrotrace.incart_fp_forensics/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "development_only",
        "evidence_note": EVIDENCE_NOTE,
        "git_head": git_head(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "metadata": model.metadata.to_dict(),
        },
        "input_hashes": input_hashes,
        "protocol": {
            "dataset": "PhysioNet INCART",
            "channel": 0,
            "polarity": "adaptive",
            "recovery": False,
            "scale_method": args.scale_method,
            "tolerance_ms": args.tolerance_ms,
            "operating_threshold": threshold,
            "retraining": False,
            "threshold_selection": "none; oracle thresholds are diagnostic only",
            "overdetect_ratio": args.overdetect_ratio,
            "annotation_policy": args.annotation_policy,
            "category_bands_ms": DEFAULT_BANDS_MS,
            "category_note": (
                "Timing bands are triage heuristics relative to the preceding/next reference beat; "
                "they do not establish a physiological cause."
            ),
        },
        "annotation_audit": {
            "summary": summarize_audits(audits),
            "records": [a.to_dict() for a in audits],
        },
        "groups": {
            "overdetecting": group_summary(over_rows, results),
            "other": group_summary(other_rows, results),
            "all": group_summary(rows, results),
        },
        "records": rows,
        "skipped_records": skipped,
        "outputs": {
            "detail_csv": csv_path.name,
            "figure": png_path.name if plotted else None,
        },
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")

    print("\n===== SUMMARY (development data) =====")
    for label in ("all", "overdetecting", "other"):
        g = report["groups"][label]
        if not g.get("n_records"):
            continue
        pooled = g["pooled_false_positives"]
        print(
            f"{label}: {g['n_records']} records, mean F1 {g['mean_f1_at_model_threshold']:.3f}, "
            f"sens {g['mean_sensitivity']:.3f}, PPV {g['mean_ppv']:.3f}, "
            f"median Stage-2 AUC {g['median_stage2_auc']}, mean oracle-threshold F1 {g['mean_oracle_best_f1']:.3f}"
        )
        fractions = pooled.get("category_fractions")
        if fractions:
            ordered = sorted(fractions.items(), key=lambda kv: -kv[1])
            print("   false positives by timing category: " + ", ".join(f"{k} {v:.0%}" for k, v in ordered if v > 0))
    print("\nWritten:", json_path)
    print("Skipped/excluded:", len(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
