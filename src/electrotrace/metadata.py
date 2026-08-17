"""Cheap recording metadata discovery without materializing full signals."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .formats import _unique_labels


def _csv_metadata(path: Path) -> dict[str, Any]:
    header = pd.read_csv(path, nrows=0)
    if header.empty:
        raise ValueError("CSV has no columns")
    from .io import TIME_COL_CANDIDATES, guess_time_column
    time_col = guess_time_column(header.columns)
    if time_col is None:
        raise ValueError("No time column found")
    signal_cols = [str(c) for c in header.columns if str(c) != str(time_col)]
    if not signal_cols:
        raise ValueError("At least one signal channel is required")
    chunks = []
    first = pd.read_csv(path, usecols=[time_col], nrows=2)
    if first.empty:
        raise ValueError("CSV contains no samples")
    last = None
    for chunk in pd.read_csv(path, usecols=[time_col], chunksize=100_000):
        last = chunk
        values = pd.to_numeric(chunk[time_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (len(values) > 1 and np.any(np.diff(values) <= 0)):
            raise ValueError("CSV time values must be finite and strictly increasing")
        chunks.append(values)
    times = np.concatenate(chunks)
    if len(times) < 2:
        raise ValueError("Recording must contain at least two samples")
    diffs = np.diff(times)
    fs = round(1.0 / float(np.median(diffs)), 6)
    return {"source_format": "csv", "sampling_rate_hz": fs, "n_samples": int(len(times)), "time_start_s": float(times[0]), "time_end_s": float(times[-1]), "duration_s": float(times[-1] - times[0]), "channels": signal_cols}


def _edf_metadata(path: Path) -> dict[str, Any]:
    try:
        import pyedflib
    except ImportError as exc:
        raise RuntimeError("EDF support requires pyedflib") from exc
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = _unique_labels([str(x).strip() for x in reader.getSignalLabels()])
        rates = [float(x) for x in reader.getSampleFrequency(0),] if reader.signals_in_file else []
        if not rates or not np.isfinite(rates[0]) or rates[0] <= 0:
            raise ValueError("EDF contains an invalid sampling rate")
        all_rates = [float(reader.getSampleFrequency(i)) for i in range(reader.signals_in_file)]
        if any(abs(r - rates[0]) > 1e-9 for r in all_rates):
            raise ValueError("EDF channels have different sampling rates")
        counts = [int(x) for x in reader.getNSamples()]
        n = min(counts)
        if n < 2:
            raise ValueError("EDF contains fewer than two samples")
        fs = rates[0]
        return {"source_format": "edf", "sampling_rate_hz": fs, "n_samples": n, "time_start_s": 0.0, "time_end_s": float((n - 1) / fs), "duration_s": float((n - 1) / fs), "channels": labels}
    finally:
        reader.close()


def _wfdb_metadata(path: Path) -> dict[str, Any]:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("WFDB support requires wfdb") from exc
    header = wfdb.rdheader(str(path))
    fs = float(header.fs)
    if not np.isfinite(fs) or fs <= 0 or int(header.sig_len) < 2:
        raise ValueError("WFDB record contains invalid metadata")
    return {"source_format": "wfdb", "sampling_rate_hz": fs, "n_samples": int(header.sig_len), "time_start_s": 0.0, "time_end_s": float((int(header.sig_len) - 1) / fs), "duration_s": float((int(header.sig_len) - 1) / fs), "channels": _unique_labels([str(x) for x in header.sig_name])}


def recording_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _csv_metadata(path)
    if suffix == ".edf":
        return _edf_metadata(path)
    if suffix in {".hea", ".dat", ".atr"}:
        return _wfdb_metadata(path.with_suffix(""))
    raise ValueError(f"Unsupported recording format: {suffix}")
