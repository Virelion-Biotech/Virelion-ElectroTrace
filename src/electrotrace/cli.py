"""Command-line interface for reproducible ElectroTrace analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .candidate_suppressor import CandidateSuppressor
from .detectors import discover_detectors, get_detector
from .io import load_recording
from .provenance import DatasetManifest
from .validation import DEFAULT_BEAT_SYMBOLS, RecordValidation, summarize_records, validate_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_files(path: Path) -> list[Path]:
    """Return the files belonging to one input record.
    
    WFDB records are multi-file objects; hashing only the .hea header would
    allow a changed .dat or annotation file to reuse stale batch state.
    """
    if path.suffix.lower() == ".hea":
        companions = sorted(path.parent.glob(f"{path.stem}.*"))
        return [p for p in companions if p.is_file()]
    return [path]


def _record_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for source in _record_files(path):
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _record_file_hashes(path: Path) -> dict[str, str]:
    return {
        source.name: _sha256(source)
        for source in _record_files(path)
    }


def _git_sha() -> str:
    return os.environ.get("ELECTROTRACE_GIT_SHA") or os.environ.get("GITHUB_SHA") or "unknown"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        if not fields:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _channel(record, channel: int) -> tuple[str, np.ndarray]:
    names = list(record.signals)
    idx = int(channel)
    if idx < 0 or idx >= len(names):
        raise ValueError(f"channel {channel} is out of range; available channels: {names}")
    name = names[idx]
    return name, np.asarray(record.signals[name], dtype=float)


def _load_model(path: str | None):
    return CandidateSuppressor.load(path) if path else None


def _two_stage(signal, fs, model, polarity, scale_method):
    from .validation_detectors import detect_r_peaks_two_stage

    peaks, probabilities = detect_r_peaks_two_stage(
        signal, fs, model, polarity=polarity, scale_method=scale_method
    )
    return np.asarray(peaks, dtype=int), np.asarray(probabilities, dtype=float)


def cmd_list(_args: argparse.Namespace) -> int:
    specs = discover_detectors()
    print("detector\tversion\tsource\tcitation")
    for spec in sorted(specs.values(), key=lambda item: item.name):
        print(f"{spec.name}\t{spec.version}\t{spec.source}\t{spec.citation}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    spec = get_detector(
        args.detector, model=model, polarity=args.polarity, scale_method=args.scale_method
    )
    record = load_recording(args.input)
    channel_name, signal = _channel(record, args.channel)
    if spec.name == "electrotrace-two-stage":
        peaks, probabilities = _two_stage(
            signal, record.sampling_rate_hz, model, args.polarity, args.scale_method
        )
    else:
        peaks = np.asarray(spec.detector(signal, record.sampling_rate_hz), dtype=int)
        probabilities = np.full(len(peaks), np.nan)

    rows = [
        {
            "sample_index": int(peak),
            "time_s": float(peak / record.sampling_rate_hz),
            "probability": float(prob) if np.isfinite(prob) else "",
        }
        for peak, prob in zip(peaks, probabilities)
    ]
    output = Path(args.output)
    _write_csv(output, rows, ["sample_index", "time_s", "probability"])
    manifest = DatasetManifest(
        dataset_id="single-record",
        dataset_version="1",
        source=str(Path(args.input).resolve()),
        records=(Path(args.input).name,),
        input_files=_record_file_hashes(Path(args.input)),
        detector_config={
            "detector": spec.name,
            "detector_version": spec.version,
            "channel_index": int(args.channel),
            "channel_name": channel_name,
            "polarity": args.polarity,
            "scale_method": args.scale_method,
        },
        software_version=__version__,
        software_commit=_git_sha(),
    ).to_dict()
    output.with_name(output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"detector={spec.name} channel={channel_name} peaks={len(peaks)} output={output}")
    return 0


def _default_inputs(directory: Path) -> list[Path]:
    matches = []
    for pattern in ("*.edf", "*.csv", "*.hea"):
        matches.extend(directory.rglob(pattern))
    return sorted(set(matches))


def _read_subject_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    mapping: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            record = str(row.get("record", "")).strip()
            subject = str(row.get("subject_id", "")).strip()
            if record and subject:
                mapping[record] = subject
    return mapping


def cmd_batch(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    spec = get_detector(
        args.detector, model=model, polarity=args.polarity, scale_method=args.scale_method
    )
    root = Path(args.input_dir)
    inputs = sorted(root.rglob(args.glob)) if args.glob else _default_inputs(root)
    if not inputs:
        raise SystemExit(f"no supported recordings matched under {root}")

    output = Path(args.output_dir)
    peaks_dir = output / "peaks"
    peaks_dir.mkdir(parents=True, exist_ok=True)
    subject_map = _read_subject_map(args.subject_map)
    state_path = output / "batch_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if args.resume and state_path.exists()
        else {}
    )
    results: list[dict] = []
    beats: list[dict] = []

    def run_one(path: Path):
        key = str(path.relative_to(root))
        cached = state.get(key)
        current_hash = _record_hash(path)
        peak_file_rel = Path("peaks") / path.relative_to(root).with_suffix(".csv")
        if (
            args.resume
            and cached
            and cached.get("source_sha256") == current_hash
            and (output / cached.get("peak_file", "")).exists()
        ):
            return cached, []

        record = load_recording(path)
        channel_name, signal = _channel(record, args.channel)
        if spec.name == "electrotrace-two-stage":
            peaks, probabilities = _two_stage(
                signal, record.sampling_rate_hz, model, args.polarity, args.scale_method
            )
        else:
            peaks = np.asarray(spec.detector(signal, record.sampling_rate_hz), dtype=int)
            probabilities = np.full(len(peaks), np.nan)

        peak_file = output / peak_file_rel
        beat_rows = [
            {
                "record": key,
                "sample_index": int(peak),
                "time_s": float(peak / record.sampling_rate_hz),
                "probability": float(prob) if np.isfinite(prob) else "",
            }
            for peak, prob in zip(peaks, probabilities)
        ]
        _write_csv(peak_file, beat_rows, ["record", "sample_index", "time_s", "probability"])
        row = {
            "record": key,
            "source_file": str(path.resolve()),
            "source_sha256": current_hash,
            "source_files": _record_file_hashes(path),
            "status": "ok",
            "channel_index": int(args.channel),
            "channel_name": channel_name,
            "sampling_rate_hz": float(record.sampling_rate_hz),
            "n_samples": int(len(signal)),
            "duration_s": float(len(signal) / record.sampling_rate_hz),
            "n_peaks": int(len(peaks)),
            "detector": spec.name,
            "detector_version": spec.version,
            "peak_file": str(peak_file_rel),
        }
        return row, beat_rows

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(run_one, path): path for path in inputs}
        for future in as_completed(futures):
            path = futures[future]
            try:
                row, rows = future.result()
            except Exception as exc:
                row = {
                    "record": str(path.relative_to(root)),
                    "source_file": str(path.resolve()),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "detector": spec.name,
                }
                rows = []
            results.append(row)
            beats.extend(rows)
            state[row["record"]] = row

    results.sort(key=lambda row: row["record"])
    beats.sort(key=lambda row: (row["record"], int(row["sample_index"])))
    _write_csv(output / "records.csv", results)
    _write_csv(output / "beats.csv", beats, ["record", "sample_index", "time_s", "probability"])
    _write_csv(
        output / "failures.csv",
        [row for row in results if row.get("status") == "failed"],
        ["record", "source_file", "status", "error", "detector"],
    )

    subject_rows: list[dict] = []
    successful = [row for row in results if row.get("status") == "ok"]
    if subject_map:
        missing = [row["record"] for row in successful if row["record"] not in subject_map]
        if missing:
            raise SystemExit(
                "--subject-map must contain a subject_id for every successful record; "
                f"missing {len(missing)} record(s)"
            )
        grouped: dict[str, dict] = {}
        for row in successful:
            subject = subject_map[row["record"]]
            agg = grouped.setdefault(
                subject, {"subject_id": subject, "n_records": 0, "n_beats": 0}
            )
            agg["n_records"] += 1
            agg["n_beats"] += int(row["n_peaks"])
        subject_rows = sorted(grouped.values(), key=lambda row: row["subject_id"])
    _write_csv(output / "subjects.csv", subject_rows, ["subject_id", "n_records", "n_beats"])
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    input_files = {}
    for row in successful:
        for name, digest in row["source_files"].items():
            input_files[f"{row['record']}::{name}"] = digest
    record_ids = tuple(sorted(row["record"] for row in successful))
    record_subject_map = (
        {record: subject_map[record] for record in record_ids}
        if subject_map
        else {}
    )
    manifest = DatasetManifest(
        dataset_id=root.name,
        dataset_version="1",
        source=str(root.resolve()),
        records=record_ids,
        subject_ids=tuple(sorted(set(record_subject_map.values()))),
        record_subject_map=record_subject_map,
        input_files=input_files,
        detector_config={
            "detector": spec.name,
            "detector_version": spec.version,
            "channel": int(args.channel),
            "polarity": args.polarity,
            "scale_method": args.scale_method,
        },
        software_version=__version__,
        software_commit=_git_sha(),
    ).to_dict()
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = sum(row.get("status") == "failed" for row in results)
    print(f"processed={len(results) - failed} failed={failed} beats={len(beats)} output={output}")
    return 0 if failed == 0 else 2


def cmd_validate(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    spec = get_detector(
        args.detector, model=model, polarity=args.polarity, scale_method=args.scale_method
    )
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_BEAT_SYMBOLS
    result = validate_record(
        args.record,
        spec.detector,
        channel=args.channel,
        annotation_extension=args.annotation,
        beat_symbols=symbols,
        tolerance_ms=args.tolerance_ms,
    )
    payload = {
        "protocol": {
            "tolerance_ms": args.tolerance_ms,
            "channel": args.channel,
            "annotation_extension": args.annotation,
            "detector": spec.name,
            "detector_version": spec.version,
            "git_sha": _git_sha(),
        },
        "record": result.to_dict(),
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["record"], sort_keys=True))
    return 0


def _bootstrap_micro_f1(
    results: list[RecordValidation], seed: int, n_boot: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(max(1, int(n_boot))):
        sample = [results[int(index)] for index in rng.integers(0, len(results), size=len(results))]
        values.append(float(summarize_records(sample)["f1"]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def cmd_bench(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    names = (
        [name.strip() for name in args.detectors.split(",") if name.strip()]
        if args.detectors
        else []
    )
    if args.all_detectors:
        names = sorted(
            discover_detectors(
                model=model, polarity=args.polarity, scale_method=args.scale_method
            )
        )
    if not names:
        raise SystemExit("provide --detectors or --all-detectors")

    records = sorted(Path(args.input_dir).rglob("*.hea"))
    if not records:
        raise SystemExit("bench currently accepts local WFDB records (*.hea)")

    summaries = []
    rows = []
    failures = []
    for name in names:
        try:
            spec = get_detector(
                name, model=model, polarity=args.polarity, scale_method=args.scale_method
            )
        except Exception as exc:
            failures.append({"detector": name, "error": str(exc)})
            continue
        detector_results = []
        for record in records:
            try:
                detector_results.append(
                    validate_record(
                        str(record.with_suffix("")),
                        spec.detector,
                        channel=args.channel,
                        annotation_extension=args.annotation,
                        beat_symbols=DEFAULT_BEAT_SYMBOLS,
                        tolerance_ms=args.tolerance_ms,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "detector": spec.name,
                        "record": record.stem,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not detector_results:
            continue
        summary = summarize_records(detector_results)
        low, high = _bootstrap_micro_f1(
            detector_results, seed=args.seed, n_boot=args.bootstrap
        )
        summary.update(
            {
                "detector": spec.name,
                "detector_version": spec.version,
                "citation": spec.citation,
                "f1_bootstrap_95ci": [low, high],
                "n_records": len(detector_results),
            }
        )
        summaries.append(summary)
        for result in detector_results:
            row = result.to_dict()
            row["detector"] = spec.name
            row["detector_version"] = spec.version
            rows.append(row)

    input_files = {}
    for record in records:
        record_path = record
        for name, digest in _record_file_hashes(record_path).items():
            input_files[f"{record_path.stem}::{name}"] = digest

    manifest = DatasetManifest(
        dataset_id=Path(args.input_dir).name,
        dataset_version="1",
        source=str(Path(args.input_dir).resolve()),
        records=tuple(record.stem for record in records),
        input_files=input_files,
        detector_config={
            "detectors": [row["detector"] for row in summaries],
            "tolerance_ms": args.tolerance_ms,
            "channel": args.channel,
            "annotation": args.annotation,
            "seed": args.seed,
            "bootstrap": args.bootstrap,
        },
        software_version=__version__,
        software_commit=_git_sha(),
    ).to_dict()
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    payload = {
        "protocol": {
            "tolerance_ms": args.tolerance_ms,
            "channel": args.channel,
            "annotation": args.annotation,
            "seed": args.seed,
            "bootstrap": args.bootstrap,
            "git_sha": _git_sha(),
        },
        "manifest": manifest,
        "summaries": summaries,
        "records": rows,
        "failures": failures,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"detectors={len(summaries)} failures={len(failures)} output={args.output}")
    return 0 if not failures else 2


def cmd_report(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    title = html.escape(args.title)
    body = html.escape(json.dumps(data, indent=2, sort_keys=True))
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        + title
        + "</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}"
        "pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem;border-radius:8px}"
        "footer{margin-top:2rem;color:#666}"
        "</style></head><body><h1>"
        + title
        + "</h1><pre>"
        + body
        + "</pre><footer>ElectroTrace "
        + html.escape(__version__)
        + " · generated "
        + datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        + "</footer></body></html>"
    )
    Path(args.output).write_text(page, encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="electrotrace",
        description="Reproducible ECG/electrophysiology analysis and benchmarking",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list available detector plugins").set_defaults(func=cmd_list)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--channel", type=int, default=0)
    common.add_argument(
        "--polarity", choices=["positive", "negative", "adaptive"], default="adaptive"
    )
    common.add_argument(
        "--scale-method",
        choices=["std", "mad", "windowed_mad", "windowed_std", "adaptive"],
        default="windowed_std",
    )
    common.add_argument("--model")

    detect = sub.add_parser("detect", parents=[common], help="detect R peaks in one recording")
    detect.add_argument("input")
    detect.add_argument("--detector", required=True)
    detect.add_argument("-o", "--output", default="peaks.csv")
    detect.set_defaults(func=cmd_detect)

    batch = sub.add_parser(
        "batch", parents=[common], help="process a directory into tidy analysis tables"
    )
    batch.add_argument("input_dir")
    batch.add_argument("--glob")
    batch.add_argument("--workers", type=int, default=1)
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--subject-map")
    batch.add_argument("-o", "--output-dir", default="electrotrace_batch")
    batch.add_argument("--detector", required=True)
    batch.set_defaults(func=cmd_batch)

    validate = sub.add_parser(
        "validate", parents=[common], help="validate a local WFDB record"
    )
    validate.add_argument("record")
    validate.add_argument("--detector", required=True)
    validate.add_argument("--annotation", default="atr")
    validate.add_argument("--symbols")
    validate.add_argument("--tolerance-ms", type=float, default=75.0)
    validate.add_argument("-o", "--output", default="validation.json")
    validate.set_defaults(func=cmd_validate)

    bench = sub.add_parser(
        "bench", parents=[common], help="benchmark detectors on local WFDB records"
    )
    bench.add_argument("input_dir")
    bench.add_argument("--detectors")
    bench.add_argument("--all-detectors", action="store_true")
    bench.add_argument("--annotation", default="atr")
    bench.add_argument("--tolerance-ms", type=float, default=75.0)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--bootstrap", type=int, default=2000)
    bench.add_argument("-o", "--output", default="bench.json")
    bench.set_defaults(func=cmd_bench)

    report = sub.add_parser(
        "report", help="render a JSON artifact as self-contained HTML"
    )
    report.add_argument("input")
    report.add_argument("-o", "--output", default="report.html")
    report.add_argument("--title", default="ElectroTrace report")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
