"""Bounded, native window access for large electrophysiology recordings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _validate_window(start: int, stop: int) -> tuple[int, int]:
    start, stop = int(start), int(stop)
    if start < 0 or stop <= start:
        raise ValueError("window must satisfy 0 <= start < stop")
    return start, stop


def read_edf_window(path: str | Path, start: int, stop: int) -> dict[str, Any]:
    start, stop = _validate_window(start, stop)
    try:
        import pyedflib
    except ImportError as exc:
        raise RuntimeError("EDF window access requires pyedflib") from exc
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = [str(x).strip() or f"channel_{i + 1}" for i, x in enumerate(reader.getSignalLabels())]
        rates = [float(reader.getSampleFrequency(i)) for i in range(reader.signals_in_file)]
        if not rates or any(not np.isfinite(r) or r <= 0 for r in rates):
            raise ValueError("EDF contains an invalid sampling rate")
        if any(abs(r - rates[0]) > 1e-9 for r in rates):
            raise ValueError("EDF channels have different sampling rates")
        n_total = int(reader.getNSamples()[0])
        if start >= n_total:
            raise ValueError("window start is beyond the recording")
        stop = min(stop, n_total)
        n = stop - start
        signals = {}
        for i, label in enumerate(labels):
            data = np.asarray(reader.readSignal(i, start=start, n=n), dtype=float)
            if len(data) != n:
                raise ValueError("EDF returned an unexpected window length")
            if not np.isfinite(data).all():
                raise ValueError(f"EDF channel '{label}' contains NaN or infinite values")
            base = label
            suffix = 2
            while label in signals:
                label = f"{base}_{suffix}"
                suffix += 1
            signals[label] = data
        fs = rates[0]
        time = np.arange(start, stop, dtype=float) / fs
        return {
            "start": start,
            "stop": stop,
            "n_samples": n_total,
            "sampling_rate_hz": fs,
            "time_start_s": 0.0,
            "time_end_s": float((n_total - 1) / fs),
            "time": time,
            "signals": signals,
        }
    finally:
        reader.close()


def read_wfdb_window(path: str | Path, start: int, stop: int) -> dict[str, Any]:
    start, stop = _validate_window(start, stop)
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("WFDB window access requires wfdb") from exc
    record = wfdb.rdrecord(str(path), sampfrom=start, sampto=stop)
    fs = float(record.fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("WFDB record contains an invalid sampling rate")
    signals = np.asarray(record.p_signal if record.p_signal is not None else record.d_signal, dtype=float)
    if signals.ndim != 2 or signals.shape[0] != stop - start:
        raise ValueError("WFDB returned an unexpected window length")
    if not np.isfinite(signals).all():
        raise ValueError("WFDB window contains NaN or infinite values")
    labels: list[str] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(record.sig_name):
        base = str(raw).strip() or f"channel_{i + 1}"
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    if len(labels) != signals.shape[1]:
        raise ValueError("WFDB channel metadata does not match signal dimensions")
    n_total = int(wfdb.rdrecord(str(path), sampfrom=0, sampto=1).sig_len)
    # sig_len from a one-sample read is not the full record length on all WFDB backends;
    # use the header metadata for the canonical total length.
    header = wfdb.rdheader(str(path))
    n_total = int(header.sig_len)
    time = np.arange(start, stop, dtype=float) / fs
    return {
        "start": start,
        "stop": stop,
        "n_samples": n_total,
        "sampling_rate_hz": fs,
        "time_start_s": 0.0,
        "time_end_s": float((n_total - 1) / fs),
        "time": time,
        "signals": {label: signals[:, i] for i, label in enumerate(labels)},
    }


def read_recording_window(path: str | Path, start: int, stop: int) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".edf":
        return read_edf_window(path, start, stop)
    if suffix in {".hea", ".dat", ".atr"}:
        return read_wfdb_window(path.with_suffix(""), start, stop)
    if suffix == ".csv":
        import pandas as pd
        start, stop = _validate_window(start, stop)
        frame = pd.read_csv(path, skiprows=lambda i: i != 0 and not (start + 1 <= i <= stop))
        if frame.empty:
            raise ValueError("window start is beyond the recording")
        return {
            "start": start,
            "stop": start + len(frame),
            "n_samples": None,
            "sampling_rate_hz": None,
            "time_start_s": float(frame.iloc[0, 0]),
            "time_end_s": float(frame.iloc[-1, 0]),
            "time": frame.iloc[:, 0].to_numpy(dtype=float),
            "signals": {str(c): pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float) for c in frame.columns[1:]},
        }
    raise ValueError(f"Unsupported windowed format: {path.suffix}")
