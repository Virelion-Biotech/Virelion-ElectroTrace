"""Native electrophysiology import for EDF and zipped WFDB records."""
from __future__ import annotations

from io import BytesIO
import os
import tempfile
import zipfile

import numpy as np


def _payload(times, signals: np.ndarray, channel_names: list[str], source_format: str) -> dict:
    signals = np.asarray(signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[:, None]
    fs = None
    if len(times) > 1:
        dt = float(np.median(np.diff(times)))
        if dt > 0:
            fs = round(1.0 / dt, 6)
    return {
        "time": np.asarray(times, dtype=float).tolist(),
        "signals": {name: signals[:, i].tolist() for i, name in enumerate(channel_names)},
        "signal_cols": channel_names,
        "sampling_rate_hz": fs,
        "duration_s": float(times[-1] - times[0]) if len(times) else 0.0,
        "n_samples": int(len(times)),
        "source_format": source_format,
    }


def load_edf(file_bytes: bytes) -> dict:
    try:
        import pyedflib
    except ImportError as exc:
        raise RuntimeError("EDF import requires pyedflib") from exc
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        reader = pyedflib.EdfReader(path)
        try:
            channels = list(reader.getSignalLabels())
            arrays = [reader.readSignal(i) for i in range(reader.signals_in_file)]
            rates = [float(reader.getSampleFrequency(i)) for i in range(reader.signals_in_file)]
            if not arrays:
                raise ValueError("EDF contains no signal channels")
            # Normalize to the first channel's grid when channels are sampled identically.
            fs = rates[0]
            if any(abs(r - fs) > 1e-9 for r in rates):
                raise ValueError("EDF channels have different sampling rates; choose/export a common-rate channel first")
            n = min(len(x) for x in arrays)
            signals = np.column_stack([np.asarray(x[:n], dtype=float) for x in arrays])
            time = np.arange(n, dtype=float) / fs
            return _payload(time, signals, channels, "EDF")
        finally:
            reader.close()
    finally:
        os.unlink(path)


def load_wfdb_zip(file_bytes: bytes) -> dict:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("WFDB import requires wfdb") from exc
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            archive.extractall(tmp)
        headers = []
        for root, _, files in os.walk(tmp):
            headers.extend(os.path.join(root, f) for f in files if f.lower().endswith(".hea"))
        if not headers:
            raise ValueError("WFDB ZIP must contain a .hea header and matching signal files")
        header = headers[0]
        record_path = os.path.splitext(header)[0]
        record = wfdb.rdrecord(record_path)
        fs = float(record.fs)
        signals = np.asarray(record.p_signal if record.p_signal is not None else record.d_signal, dtype=float)
        channels = [str(x) for x in record.sig_name]
        time = np.arange(signals.shape[0], dtype=float) / fs
        return _payload(time, signals, channels, "WFDB")


def load_electrophysiology(file_bytes: bytes, filename: str) -> dict:
    suffix = os.path.splitext(filename.lower())[1]
    if suffix == ".edf":
        return load_edf(file_bytes)
    if suffix in {".zip", ".wfdb"}:
        return load_wfdb_zip(file_bytes)
    raise ValueError("Supported native formats are .edf and WFDB .zip archives")
