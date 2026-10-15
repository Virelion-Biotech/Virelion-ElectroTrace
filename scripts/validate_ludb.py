"""LUDB paired-boundary QRS characterization with explicit reference exclusions.

The frozen delineator receives manual QRS centers (oracle-assisted). Each N
marker is counted. Only adjacent, ordered ( N ) references are eligible; missing,
reordered or ambiguous boundaries are reported, never imputed. P/T performance,
R-peak detection and lead-selector qualification are outside this protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from scripts import validate_qtdb as qtdb

LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
TOLERANCES = [20.0, 40.0, 60.0, 80.0, 100.0]
PARSER_VERSION = "adjacent-complete-qrs-v2"


def qrs_reference(annotation, signal_length):
    """Extract conservatively scorable QRS pairs and account for every N marker.

    Nonadjacent boundaries may reflect overlap or incomplete labels. Excluding
    them limits the estimand to complete adjacent reference triplets; it does
    not assert their physiology is invalid or their missing values are zero.
    """
    samples = np.asarray(annotation.sample)
    symbols = list(annotation.symbol)
    if samples.ndim != 1 or len(samples) != len(symbols):
        raise ValueError("Annotation sample/symbol length mismatch")
    if not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("Annotation samples must be integers")
    if np.any(samples < 0) or np.any(samples >= signal_length) or np.any(np.diff(samples) < 0):
        raise ValueError("Annotation sample outside signal or not monotonic")
    if set(symbols) - {"(", ")", "p", "N", "t"}:
        raise ValueError("Unsupported LUDB annotation symbol")
    pairs, excluded = [], []
    for i, symbol in enumerate(symbols):
        if symbol != "N":
            continue
        reasons = []
        if i == 0 or symbols[i - 1] != "(":
            reasons.append("missing_or_nonadjacent_onset")
        if i + 1 == len(symbols) or symbols[i + 1] != ")":
            reasons.append("missing_or_nonadjacent_offset")
        if not reasons and not samples[i - 1] < samples[i + 1]:
            reasons.append("nonpositive_support")
        if reasons:
            excluded.append({"annotation_index": i, "peak_sample": int(samples[i]), "reasons": reasons})
        else:
            pairs.append((int(samples[i - 1]), int(samples[i + 1])))
    if not pairs and not excluded:
        raise ValueError("No QRS markers")
    boundaries = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(boundaries) > 1 and (np.any(np.diff(boundaries[:, 0]) <= 0) or np.any(np.diff(boundaries[:, 1]) <= 0)):
        raise ValueError("Duplicate/nonmonotonic QRS pairs")
    audit = {
        "qrs_markers": len(pairs) + len(excluded),
        "eligible_paired_qrs": len(pairs),
        "excluded_qrs": len(excluded),
        "excluded": excluded,
        "parser_version": PARSER_VERSION,
    }
    return boundaries[:, 0], boundaries[:, 1], audit


def patient_result(wfdb, path):
    rec = wfdb.rdrecord(str(path))
    if rec.fs != 500 or rec.p_signal.shape != (5000, 12):
        raise ValueError("Unexpected LUDB record sampling/shape")
    if not np.isfinite(rec.p_signal).all():
        raise ValueError("Nonfinite LUDB signal")
    channels = [x.lower() for x in rec.sig_name]
    if sorted(channels) != sorted(LEADS):
        raise ValueError("Missing/duplicate LUDB leads")
    perlead = {}
    for lead in LEADS:
        signal = rec.p_signal[:, channels.index(lead)]
        on, off, audit = qrs_reference(wfdb.rdann(str(path), lead), len(signal))
        err = qtdb.delineation_errors(signal, rec.fs, (on + off) // 2, on, off)
        perlead[lead] = {
            "reference_audit": audit,
            "primary": {
                "reference_qrs": len(on),
                "onset_found": int(err["onset_found"].sum()),
                "offset_found": int(err["offset_found"].sum()),
                "both_found": int((err["onset_found"] & err["offset_found"]).sum()),
                "tolerance_counts": qtdb.tolerance_counts(
                    err["onset_errors_ms"], err["offset_errors_ms"], err["onset_found"], err["offset_found"], TOLERANCES
                ),
            },
            "onset_error_ms": err["onset_errors_ms"].tolist(),
            "offset_error_ms": err["offset_errors_ms"].tolist(),
        }
    return {"record": "data/" + path.name, "leads": perlead}


def summarize(rows):
    if len(rows) != 200 or len({x["record"] for x in rows}) != 200:
        raise ValueError("Summary requires all 200 distinct LUDB patients")
    opts = {"replicates": 2000, "seed": 1729, "interval": (2.5, 97.5)}
    pooled = []
    audits = []
    for row in rows:
        vals = [row["leads"][lead]["primary"] for lead in LEADS]
        audits.extend(row["leads"][lead]["reference_audit"] for lead in LEADS)
        primary = {k: sum(v[k] for v in vals) for k in ["reference_qrs", "onset_found", "offset_found", "both_found"]}
        primary["tolerance_counts"] = {
            key: {k: sum(v["tolerance_counts"][key][k] for v in vals) for k in vals[0]["tolerance_counts"][key]}
            for key in vals[0]["tolerance_counts"]
        }
        pooled.append({"primary": primary})
    totals = {k: sum(a[k] for a in audits) for k in ["qrs_markers", "eligible_paired_qrs", "excluded_qrs"]}
    return {
        "status": "COMPLETE_SCOPED",
        "n_patients": 200,
        "n_leads": 2400,
        "oracle_assisted": True,
        "parser_version": PARSER_VERSION,
        "reference_coverage": totals,
        "estimand": "Boundary tolerance success among complete adjacent manually labeled QRS triplets",
        "bootstrap_unit": "patient (all twelve leads retained together)",
        "per_lead": {
            lead: qtdb.aggregate_primary_curve([x["leads"][lead] for x in rows], TOLERANCES, **opts) for lead in LEADS
        },
        "all_leads_patient_clustered": qtdb.aggregate_primary_curve(pooled, TOLERANCES, **opts),
        "lead_selector_qualification": False,
        "end_to_end_detection_evaluated": False,
        "limitations": [
            "Public dataset with unproven absence of prior exposure: exploratory characterization.",
            "Reference exclusions are counted, not imputed; results do not cover all marked QRS.",
            "No P/T, device transport, independent-cohort or clinical qualification.",
        ],
    }


def atomic_json(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=7200)
    args = parser.parse_args()
    import wfdb

    manifest = args.data / "SHA256SUMS.txt"
    if sha(manifest) != "cccef1f3529519db8f26a333c97a6872a7d0e3e5c64a1448b76b161fd87fb75f":
        raise ValueError("LUDB manifest mismatch")
    lookup = {p.lstrip("*"): h for h, p in map(str.split, manifest.read_text().splitlines())}
    if sha(args.data / "RECORDS") != lookup["RECORDS"]:
        raise ValueError("LUDB RECORDS mismatch")
    names = (args.data / "RECORDS").read_text().split()
    if len(names) != 200 or len(set(names)) != 200:
        raise ValueError("Expected 200 distinct patients")
    root = Path(__file__).resolve().parents[1]
    fingerprints = {
        str(p.relative_to(root)): sha(p)
        for p in [Path(__file__).resolve(), root / "scripts/validate_qtdb.py", *sorted((root / "src").rglob("*.py"))]
    }
    identity = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    protocol = {
        "parser_version": PARSER_VERSION,
        "source_sha256": fingerprints,
        "source_identity": identity,
        "manifest_sha256": sha(manifest),
        "tolerances_ms": TOLERANCES,
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 1729,
        "estimand": "Complete adjacent QRS triplets; exclusions audited; manual centers",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "wfdb": wfdb.__version__,
        },
    }
    old = args.output / "protocol.json"
    if old.exists() and json.loads(old.read_text()) != protocol:
        raise ValueError("Checkpoint protocol/environment differs; choose a new output folder")
    atomic_json(old, protocol)
    started = time.monotonic()
    rows = []
    for name in names:
        if time.monotonic() - started >= args.budget_seconds:
            atomic_json(args.output / "RUN_STATUS.json", {"status": "PAUSED", "completed_patients": len(rows)})
            return
        if not name.startswith("data/") or not name[5:].isdigit():
            raise ValueError("Unsafe record path")
        receipts = {}
        for ext in ["hea", "dat", *LEADS]:
            path = args.data / (name + "." + ext)
            if sha(path) != lookup[name + "." + ext]:
                raise ValueError(f"Input hash mismatch: {path.name}")
            receipts[name + "." + ext] = lookup[name + "." + ext]
        dest = args.output / "patients" / (Path(name).name + ".json")
        checkpoint = args.output / "state" / (Path(name).name + ".json")
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if state != {"identity": identity, "inputs": receipts, "result_sha256": sha(dest)}:
                raise ValueError("Patient checkpoint integrity/identity mismatch")
            row = json.loads(dest.read_text())
        else:
            row = patient_result(wfdb, args.data / name)
            atomic_json(dest, row)
            atomic_json(checkpoint, {"identity": identity, "inputs": receipts, "result_sha256": sha(dest)})
        rows.append(row)
        print(f"Verified patient {len(rows)}/200", flush=True)
    atomic_json(args.output / "summary.json", summarize(rows))
    atomic_json(args.output / "RUN_STATUS.json", {"status": "COMPLETE_SCOPED", "completed_patients": 200})


if __name__ == "__main__":
    main()
