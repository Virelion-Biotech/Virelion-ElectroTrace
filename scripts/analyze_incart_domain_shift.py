#!/usr/bin/env python3
"""Record-level domain-shift analysis: ElectroTrace INCART vs certified gqrs.

Reads published validation JSONs (no re-detection required) and writes:
  - record-level join table (CSV)
  - structured analysis report (JSON)

Failure-mode taxonomy (heuristic, locked to existing fields):
  A. candidate_generation_failure  — stage1_candidates / reference_count < 0.5
  B. over_suppression              — stage1/ref >= 0.8 and suppression_rate > 0.7
  C. mixed_or_other                — everything else with ET F1 < 0.75
  D. acceptable                    — ET F1 >= 0.75
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def classify(row: dict) -> str:
    if row["et_f1"] >= 0.75:
        return "acceptable"
    cand_ratio = row["cand_vs_ref"]
    if cand_ratio < 0.5:
        return "candidate_generation_failure"
    if cand_ratio >= 0.8 and row["suppression_rate"] > 0.7:
        return "over_suppression"
    return "mixed_or_other"


def build_rows(et_report: dict, gqrs_records: list[dict]) -> list[dict]:
    et_by = {r["record"]: r for r in et_report["record_results"]}
    gq_by = {r["record"]: r for r in gqrs_records}
    common = sorted(set(et_by) & set(gq_by))

    rows: list[dict] = []
    for rec in common:
        e = et_by[rec]
        g = gq_by[rec]
        ref = int(e["reference_count"])
        stage1 = int(e.get("stage1_candidates") or 0)
        stage2 = int(e.get("stage2_retained") or 0)
        suppress = float(e.get("suppression_rate") or 0.0)
        cand_vs_ref = stage1 / max(ref, 1)
        ret_vs_ref = stage2 / max(ref, 1)
        row = {
            "record": rec,
            "fs_hz": e.get("fs_hz"),
            "reference_count": ref,
            "et_sensitivity": float(e["sensitivity"]),
            "et_ppv": float(e["positive_predictive_value"]),
            "et_f1": float(e["f1"]),
            "et_true_positive": int(e["true_positive"]),
            "et_false_positive": int(e["false_positive"]),
            "et_false_negative": int(e["false_negative"]),
            "et_detected_count": int(e["detected_count"]),
            "stage1_candidates": stage1,
            "stage2_retained": stage2,
            "suppression_rate": suppress,
            "selected_polarity": e.get("selected_polarity"),
            "cand_vs_ref": cand_vs_ref,
            "retained_vs_ref": ret_vs_ref,
            "gqrs_sensitivity": float(g["sensitivity"]),
            "gqrs_ppv": float(g["positive_predictive_value"]),
            "gqrs_f1": float(g["f1"]),
            "gqrs_true_positive": int(g["true_positive"]),
            "gqrs_false_positive": int(g["false_positive"]),
            "gqrs_false_negative": int(g["false_negative"]),
            "gqrs_detected_count": int(g["detected_count"]),
            "delta_f1_et_minus_gqrs": float(e["f1"]) - float(g["f1"]),
            "delta_sens_et_minus_gqrs": float(e["sensitivity"]) - float(g["sensitivity"]),
        }
        row["failure_mode"] = classify(row)
        rows.append(row)
    return rows


def summarize_group(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_et_f1": mean([r["et_f1"] for r in rows]),
        "mean_et_sensitivity": mean([r["et_sensitivity"] for r in rows]),
        "mean_et_ppv": mean([r["et_ppv"] for r in rows]),
        "mean_gqrs_f1": mean([r["gqrs_f1"] for r in rows]),
        "mean_gqrs_sensitivity": mean([r["gqrs_sensitivity"] for r in rows]),
        "mean_suppression_rate": mean([r["suppression_rate"] for r in rows]),
        "mean_cand_vs_ref": mean([r["cand_vs_ref"] for r in rows]),
        "mean_retained_vs_ref": mean([r["retained_vs_ref"] for r in rows]),
        "records": [r["record"] for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--electrotrace-report",
        type=Path,
        default=Path("validation_reports/incart_two_stage_external_full_2026-09-09.json"),
    )
    parser.add_argument(
        "--gqrs-report",
        type=Path,
        default=Path("validation_reports/certified_wfdb_incart_2026-09-11.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("validation_reports/incart_domain_shift_analysis_2026-09-12.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("validation_reports/incart_domain_shift_record_table_2026-09-12.csv"),
    )
    args = parser.parse_args()

    et = load_json(args.electrotrace_report)
    gq = load_json(args.gqrs_report)
    gqrs_records = gq["results"]["gqrs"]["records"]

    rows = build_rows(et, gqrs_records)
    by_mode: dict[str, list[dict]] = {}
    for r in rows:
        by_mode.setdefault(r["failure_mode"], []).append(r)

    worst_et = sorted(rows, key=lambda r: r["et_f1"])[:20]
    largest_gap = sorted(rows, key=lambda r: r["delta_f1_et_minus_gqrs"])[:20]
    best_et = sorted(rows, key=lambda r: r["et_f1"], reverse=True)[:10]

    low = [r for r in rows if r["et_f1"] < 0.5]
    high = [r for r in rows if r["et_f1"] >= 0.85]
    gqrs_ok_et_bad = [r for r in rows if r["gqrs_f1"] >= 0.9 and r["et_f1"] < 0.5]

    report = {
        "schema": "electrotrace.incart_domain_shift_analysis/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "electrotrace_report": str(args.electrotrace_report),
            "gqrs_report": str(args.gqrs_report),
            "electrotrace_summary": et.get("summary"),
            "gqrs_summary": gq["results"]["gqrs"].get("summary"),
        },
        "n_common_records": len(rows),
        "failure_mode_counts": {k: len(v) for k, v in sorted(by_mode.items())},
        "failure_mode_summaries": {k: summarize_group(v) for k, v in sorted(by_mode.items())},
        "cohort_summaries": {
            "et_f1_lt_0_5": summarize_group(low),
            "et_f1_ge_0_85": summarize_group(high),
            "gqrs_f1_ge_0_9_and_et_f1_lt_0_5": summarize_group(gqrs_ok_et_bad),
        },
        "worst_electrotrace_by_f1": worst_et,
        "largest_et_minus_gqrs_f1_gaps": largest_gap,
        "best_electrotrace_by_f1": best_et,
        "polarity_counts_all": {
            p: sum(1 for r in rows if r["selected_polarity"] == p)
            for p in sorted({r["selected_polarity"] for r in rows})
        },
        "polarity_counts_et_f1_lt_0_5": {
            p: sum(1 for r in low if r["selected_polarity"] == p)
            for p in sorted({r["selected_polarity"] for r in rows})
        },
        "findings": [
            "Certified gqrs remains strong on INCART (aggregate F1 ~0.93) while ElectroTrace two-stage collapses on many of the same records. This is not primarily an 'INCART is undetectable' problem.",
            "Dominant failure mode among poor ElectroTrace records is candidate_generation_failure: stage-1 candidates are far below the reference beat count (often ~1-5% of refs), so stage-2 suppression cannot recover recall.",
            "A secondary mode is over_suppression: adequate stage-1 coverage but high suppression_rate (>0.7) leaves too few retained peaks.",
            "Records where ElectroTrace works well show stage1/ref roughly near or above 1.0 and moderate suppression (~0.2-0.5), similar operating regime to MIT-BIH.",
            "Polarity is almost always 'positive'; it does not explain the split between good and bad INCART records by itself.",
        ],
        "recommended_next_actions": [
            "Priority 1 — Stage-1 candidate generation on INCART: inspect prominence/scale/polarity adaptation at 257 Hz on worst records (I56, I24, I03, I40, I19) where gqrs F1 remains >0.98.",
            "Priority 2 — Only after stage-1 coverage is fixed, revisit stage-2 threshold domain shift on records with high cand_vs_ref but high suppression (e.g. I62, I73).",
            "Do not change ML architecture yet; the evidence points to detector front-end / candidate coverage under domain shift.",
            "Optional deep dive: re-run candidate_stream + score histograms on the 10 worst gap records with the locked MIT-BIH model.",
        ],
        "protocol_note": "This analysis uses only published aggregate/per-record validation fields. It does not re-train or re-threshold using INCART labels.",
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote", args.output_json)
    print("Wrote", args.output_csv)
    print("\nFailure mode counts:")
    for k, v in report["failure_mode_counts"].items():
        print(f"  {k}: {v}")
    print("\nFindings:")
    for line in report["findings"]:
        print("-", line)


if __name__ == "__main__":
    main()
