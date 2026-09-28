#!/usr/bin/env python3
"""Develop label-free two-lead selection rules on exposed EDB data.

This is explicitly post-hoc development analysis. EDB labels are used only to
compare a small prespecified family of selectors after each selector has chosen
a lead from label-free detector/signal summaries. The selected rule must be
frozen and evaluated on a different untouched database before any prospective
generalization claim.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from electrotrace.candidate_suppressor import CandidateSuppressor

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_posthoc_helpers():
    root = str(REPO_ROOT)
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        from scripts import analyze_edb_posthoc_failures as posthoc
        return posthoc
    finally:
        if added:
            sys.path.remove(root)


RULE_ORDER = (
    "retained_qrs_band_fraction",
    "qrs_band_x_retained_probability",
    "retained_prominence_z",
    "retained_probability_p50",
    "retention_fraction",
    "qrs_band_per_width",
)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _finite(value, default=float("-inf")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def label_free_scores(channel: dict) -> dict[str, float]:
    retained = channel.get("feature_medians_retained") or {}
    quality = channel.get("label_free_quality") or {}
    prob = quality.get("retained_probability") or {}

    qrs = _finite(retained.get("qrs_band_fraction"), 0.0)
    prom = _finite(retained.get("prominence_z"), 0.0)
    width = _finite(retained.get("width_s"), float("inf"))
    p50 = _finite(prob.get("p50"), 0.0)
    retention = _finite(quality.get("retention_fraction"), 0.0)

    return {
        "retained_qrs_band_fraction": qrs,
        "qrs_band_x_retained_probability": qrs * p50,
        "retained_prominence_z": prom,
        "retained_probability_p50": p50,
        "retention_fraction": retention,
        "qrs_band_per_width": qrs / max(width, 1e-6) if math.isfinite(width) else 0.0,
    }


def choose_channel(rule: str, ch0: dict, ch1: dict) -> int:
    s0 = label_free_scores(ch0)
    s1 = label_free_scores(ch1)
    if rule not in s0:
        raise KeyError(rule)
    # Deterministic tie behavior preserves the preregistered historical channel.
    return 1 if s1[rule] > s0[rule] else 0


def summarize_selected(rows: list[dict], key: str) -> dict:
    tp = fp = fn = 0
    f1s = []
    sens = []
    ppv = []
    n_ch1 = 0
    regressions = 0
    improvements = 0
    unchanged = 0
    for row in rows:
        selected = row["selectors"][key]
        metrics = row[f"channel{selected}"]["metrics"]
        baseline = row["channel0"]["metrics"]
        tp += int(metrics["true_positive"])
        fp += int(metrics["false_positive"])
        fn += int(metrics["false_negative"])
        f1s.append(float(metrics["f1"]))
        sens.append(float(metrics["sensitivity"]))
        ppv.append(float(metrics["positive_predictive_value"]))
        n_ch1 += int(selected == 1)
        delta = float(metrics["f1"]) - float(baseline["f1"])
        if delta > 1e-12:
            improvements += 1
        elif delta < -1e-12:
            regressions += 1
        else:
            unchanged += 1

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2.0 * sensitivity * precision / (sensitivity + precision)
        if sensitivity + precision
        else 0.0
    )
    return {
        "records": len(rows),
        "selected_channel1_records": n_ch1,
        "selected_channel0_records": len(rows) - n_ch1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity": sensitivity,
        "positive_predictive_value": precision,
        "f1": f1,
        "macro_mean_record_f1": float(np.mean(f1s)),
        "macro_median_record_f1": float(np.median(f1s)),
        "min_record_f1": float(np.min(f1s)),
        "max_record_f1": float(np.max(f1s)),
        "records_improved_vs_channel0": improvements,
        "records_regressed_vs_channel0": regressions,
        "records_unchanged_vs_channel0": unchanged,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edb-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--first-run-report", type=Path, required=True)
    ap.add_argument("--polarity-threshold-report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    posthoc = _load_posthoc_helpers()
    first_run = posthoc.load_first_run(args.first_run_report)
    model = CandidateSuppressor.load(args.model)
    model_identity = posthoc.verify_semantic_model_identity(model, first_run)
    _, width_gate, v2_gate = posthoc.load_polarity_threshold_report(
        args.polarity_threshold_report
    )

    archived_thresholds = first_run["polarity_threshold_derivation"]["recommended_thresholds"]
    if float(v2_gate) != float(archived_thresholds["v2_gate_confidence"]):
        raise SystemExit("v2 gate does not match archived prospective run")
    if float(width_gate) != float(archived_thresholds["width_override_confidence"]):
        raise SystemExit("width gate does not match archived prospective run")

    protocol = first_run["protocol"]["detector"]
    scale_method = str(protocol["stage1_scale_method"])
    tolerance_ms = float(protocol["tolerance_ms"])
    archived = {str(r["record"]): r for r in first_run["record_results"]}
    records = [str(r) for r in first_run["protocol"]["dataset"]["records"]]

    if len(records) != 90 or len(set(records)) != 90:
        raise SystemExit("archived prospective record list is not the canonical 90-record cohort")

    rows = []
    for i, record in enumerate(records, start=1):
        base = args.edb_dir / record
        header = wfdb.rdheader(str(base))
        names = list(header.sig_name or [])

        ch0 = posthoc.diagnose_channel(
            base,
            channel=0,
            model=model,
            v2_gate=v2_gate,
            width_gate=width_gate,
            scale_method=scale_method,
            tolerance_ms=tolerance_ms,
        )
        archived_f1 = float(archived[record]["f1"])
        if abs(float(ch0["metrics"]["f1"]) - archived_f1) > 1e-12:
            raise SystemExit(
                f"{record}: channel-0 F1 does not reproduce archived prospective run "
                f"({ch0['metrics']['f1']} != {archived_f1})"
            )

        ch1 = posthoc.diagnose_channel(
            base,
            channel=1,
            model=model,
            v2_gate=v2_gate,
            width_gate=width_gate,
            scale_method=scale_method,
            tolerance_ms=tolerance_ms,
        )

        selectors = {rule: choose_channel(rule, ch0, ch1) for rule in RULE_ORDER}
        selectors["channel0_baseline"] = 0
        selectors["channel1_always"] = 1
        selectors["oracle_by_f1_development_only"] = (
            1 if float(ch1["metrics"]["f1"]) > float(ch0["metrics"]["f1"]) else 0
        )

        rows.append(
            {
                "record": record,
                "signal_names": names,
                "channel0": ch0,
                "channel1": ch1,
                "label_free_scores": {
                    "channel0": label_free_scores(ch0),
                    "channel1": label_free_scores(ch1),
                },
                "selectors": selectors,
                "channel1_minus_channel0_f1": (
                    float(ch1["metrics"]["f1"]) - float(ch0["metrics"]["f1"])
                ),
            }
        )
        print(
            f"[{i:02d}/90] {record}: "
            f"ch0={ch0['metrics']['f1']:.4f} "
            f"ch1={ch1['metrics']['f1']:.4f} "
            f"qrs-rule->{selectors['retained_qrs_band_fraction']}"
        )

    summaries = {
        "channel0_baseline": summarize_selected(rows, "channel0_baseline"),
        "channel1_always": summarize_selected(rows, "channel1_always"),
        **{rule: summarize_selected(rows, rule) for rule in RULE_ORDER},
        "oracle_by_f1_development_only": summarize_selected(
            rows, "oracle_by_f1_development_only"
        ),
    }

    # This is development selection using exposed EDB labels. It is not external
    # validation. Complexity is constrained to the small RULE_ORDER family above.
    best_rule = max(
        RULE_ORDER,
        key=lambda rule: (
            summaries[rule]["f1"],
            summaries[rule]["macro_mean_record_f1"],
            -RULE_ORDER.index(rule),
        ),
    )

    regressions = []
    for row in rows:
        selected = row["selectors"][best_rule]
        chosen_f1 = float(row[f"channel{selected}"]["metrics"]["f1"])
        base_f1 = float(row["channel0"]["metrics"]["f1"])
        if chosen_f1 + 1e-12 < base_f1:
            regressions.append(
                {
                    "record": row["record"],
                    "selected_channel": selected,
                    "channel0_f1": base_f1,
                    "selected_f1": chosen_f1,
                    "delta_f1": chosen_f1 - base_f1,
                    "signal_names": row["signal_names"],
                }
            )

    report = {
        "schema": "electrotrace.edb_label_free_lead_selector_development/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "posthoc_exposed_dataset_development_analysis",
        "git_head": git_head(),
        "guardrail": (
            "EDB is exposed development data. EDB labels are used here only to compare "
            "selectors whose runtime decisions use label-free summaries. The selected "
            "rule is not prospectively validated until frozen and tested on a different "
            "untouched database."
        ),
        "inputs": {
            "first_run_report": str(args.first_run_report),
            "first_run_sha256": posthoc.sha256_file(args.first_run_report),
            "polarity_derivation": str(args.polarity_threshold_report),
            "polarity_derivation_sha256": posthoc.sha256_file(args.polarity_threshold_report),
            "model_semantic_identity": model_identity,
            "reconstructed_model_sha256": posthoc.sha256_file(args.model),
        },
        "candidate_rule_family": list(RULE_ORDER),
        "selection_method": (
            "Choose the candidate rule with highest EDB development aggregate F1; "
            "break ties by macro mean F1 then RULE_ORDER. No continuous threshold sweep."
        ),
        "rule_summaries": summaries,
        "selected_development_rule": best_rule,
        "selected_rule_summary": summaries[best_rule],
        "selected_rule_regressions_vs_channel0": regressions,
        "records": rows,
        "non_claims": [
            "The selected lead rule is EDB-informed development, not prospective evidence.",
            "Channel-1 and oracle results do not replace the archived channel-0 prospective result.",
            "A different untouched database is required before claiming generalization of automatic lead selection.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nRule summaries:")
    for name, summary in summaries.items():
        print(
            f"{name}: F1={summary['f1']:.6f}, macro={summary['macro_mean_record_f1']:.6f}, "
            f"ch1={summary['selected_channel1_records']}, regressions={summary['records_regressed_vs_channel0']}"
        )
    print("Selected development rule:", best_rule)
    print("Wrote:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
