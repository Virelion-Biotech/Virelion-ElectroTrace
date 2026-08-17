"""External ECG detector validation against PhysioNet/WFDB references."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_BEAT_SYMBOLS = frozenset({
    "N", "L", "R", "A", "a", "J", "S", "V", "E", "j", "F", "/", "f", "Q", "e",
})


@dataclass(frozen=True)
class DetectionMetrics:
    reference_count: int
    detected_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    sensitivity: float
    positive_predictive_value: float
    f1: float
    median_timing_error_ms: float | None
    p95_timing_error_ms: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RecordValidation:
    record: str
    fs_hz: float
    metrics: DetectionMetrics

    def to_dict(self) -> dict:
        return {"record": self.record, "fs_hz": self.fs_hz, **self.metrics.to_dict()}


def _validate_times(values: Iterable[float], name: str) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite one-dimensional sequence")
    if values.size and np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def match_peaks(
    detected_samples: Sequence[int],
    reference_samples: Sequence[int],
    fs_hz: float,
    tolerance_ms: float = 75.0,
) -> DetectionMetrics:
    """Match detected and reference beats one-to-one within a timing tolerance."""
    fs_hz = float(fs_hz)
    tolerance_ms = float(tolerance_ms)
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be positive and finite")
    if not np.isfinite(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError("tolerance_ms must be positive and finite")

    detected = np.asarray(detected_samples, dtype=int)
    reference = np.asarray(reference_samples, dtype=int)
    if detected.ndim != 1 or reference.ndim != 1:
        raise ValueError("detected_samples and reference_samples must be one-dimensional")
    detected = np.unique(detected[detected >= 0])
    reference = np.unique(reference[reference >= 0])
    tolerance_samples = tolerance_ms * fs_hz / 1000.0

    i = j = tp = 0
    errors: list[float] = []
    while i < len(detected) and j < len(reference):
        delta = int(detected[i]) - int(reference[j])
        if abs(delta) <= tolerance_samples:
            tp += 1
            errors.append(abs(delta) * 1000.0 / fs_hz)
            i += 1
            j += 1
        elif detected[i] < reference[j]:
            i += 1
        else:
            j += 1

    fp = len(detected) - tp
    fn = len(reference) - tp
    sensitivity = tp / len(reference) if reference.size else 0.0
    ppv = tp / len(detected) if detected.size else 0.0
    f1 = (2 * sensitivity * ppv / (sensitivity + ppv)) if sensitivity + ppv else 0.0
    median = float(np.median(errors)) if errors else None
    p95 = float(np.percentile(errors, 95)) if errors else None
    return DetectionMetrics(
        reference_count=int(len(reference)),
        detected_count=int(len(detected)),
        true_positive=int(tp),
        false_positive=int(fp),
        false_negative=int(fn),
        sensitivity=float(sensitivity),
        positive_predictive_value=float(ppv),
        f1=float(f1),
        median_timing_error_ms=median,
        p95_timing_error_ms=p95,
    )


def _record_signal(record_path: str | Path, channel: int = 0):
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("wfdb is required for PhysioNet validation") from exc
    record = wfdb.rdrecord(str(record_path), channels=[int(channel)], physical=False)
    signal = np.asarray(record.p_signal[:, 0] if record.p_signal is not None else record.d_signal[:, 0], dtype=float)
    return signal, float(record.fs)


def load_reference_annotations(record_path: str | Path, extension: str = "atr", symbols: Iterable[str] | None = None) -> np.ndarray:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("wfdb is required for PhysioNet validation") from exc
    ann = wfdb.rdann(str(record_path), extension=extension)
    allowed = set(symbols) if symbols is not None else DEFAULT_BEAT_SYMBOLS
    samples = [int(s) for s, symbol in zip(ann.sample, ann.symbol) if symbol in allowed]
    return np.asarray(samples, dtype=int)


def validate_record(
    record_path: str | Path,
    detector,
    channel: int = 0,
    annotation_extension: str = "atr",
    beat_symbols: Iterable[str] | None = None,
    tolerance_ms: float = 75.0,
) -> RecordValidation:
    """Run a detector on one WFDB record and compare with reference annotations.

    ``detector`` receives ``(signal, fs_hz)`` and must return sample indices.
    """
    signal, fs_hz = _record_signal(record_path, channel=channel)
    reference = load_reference_annotations(record_path, extension=annotation_extension, symbols=beat_symbols)
    detected = np.asarray(detector(signal, fs_hz), dtype=int)
    metrics = match_peaks(detected, reference, fs_hz, tolerance_ms=tolerance_ms)
    return RecordValidation(record=Path(record_path).stem, fs_hz=fs_hz, metrics=metrics)


def summarize_records(results: Sequence[RecordValidation]) -> dict:
    if not results:
        raise ValueError("At least one validation result is required")
    tp = sum(r.metrics.true_positive for r in results)
    fp = sum(r.metrics.false_positive for r in results)
    fn = sum(r.metrics.false_negative for r in results)
    reference_count = sum(r.metrics.reference_count for r in results)
    detected_count = sum(r.metrics.detected_count for r in results)
    sensitivity = tp / reference_count if reference_count else 0.0
    ppv = tp / detected_count if detected_count else 0.0
    f1 = (2 * sensitivity * ppv / (sensitivity + ppv)) if sensitivity + ppv else 0.0
    timing = [
        r.metrics.median_timing_error_ms
        for r in results
        if r.metrics.median_timing_error_ms is not None
    ]
    return {
        "records": len(results),
        "reference_count": reference_count,
        "detected_count": detected_count,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity": float(sensitivity),
        "positive_predictive_value": float(ppv),
        "f1": float(f1),
        "median_record_timing_error_ms": float(np.median(timing)) if timing else None,
    }


def _main() -> int:
    import argparse
    import importlib
    import json

    parser = argparse.ArgumentParser(description="Validate an ECG detector against PhysioNet/WFDB annotations.")
    parser.add_argument("records", nargs="+", help="Local WFDB record paths without .hea suffix")
    parser.add_argument("--detector", required=True, help="Python callable in module:function form")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--annotation-extension", default="atr")
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument("--symbols", help="Comma-separated reference beat symbols")
    args = parser.parse_args()

    module_name, function_name = args.detector.split(":", 1)
    detector = getattr(importlib.import_module(module_name), function_name)
    symbols = args.symbols.split(",") if args.symbols else None
    results = [
        validate_record(
            record,
            detector,
            channel=args.channel,
            annotation_extension=args.annotation_extension,
            beat_symbols=symbols,
            tolerance_ms=args.tolerance_ms,
        )
        for record in args.records
    ]
    print(json.dumps({"records": [r.to_dict() for r in results], "summary": summarize_records(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
