"""ElectroTrace local research web server."""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from scipy import signal as sps

from electrotrace.io import load_csv, validate_dataframe, load_recording
from electrotrace.signal import apply_pipeline, FilterConfigurationError
from electrotrace.beats import segment_beats
from electrotrace.phenotype import beat_phenotypes, summary_statistics
from electrotrace.statistics import compare_groups, benjamini_hochberg
from electrotrace.benchmark import benchmark_models
from electrotrace.project_store import ProjectStore, RecordingRef

WEB = os.path.join(ROOT, "web")
SAMPLE_DATA = os.path.join(ROOT, "sample_data")
UPLOAD_ROOT = Path(os.getenv("ELECTROTRACE_UPLOAD_ROOT", os.path.join(tempfile.gettempdir(), "electrotrace_uploads")))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(os.getenv("ELECTROTRACE_PROJECT_ROOT", os.path.join(ROOT, "projects")))
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder=WEB, static_url_path="")


def _save_upload(uploaded) -> tuple[str, Path]:
    name = Path(uploaded.filename or "recording.bin")
    suffix = name.suffix.lower()
    token = uuid.uuid4().hex
    destination = UPLOAD_ROOT / f"{token}{suffix}"
    uploaded.save(destination)
    if suffix == ".zip":
        extract_dir = UPLOAD_ROOT / token
        extract_dir.mkdir()
        with zipfile.ZipFile(destination) as zf:
            for member in zf.infolist():
                target = (extract_dir / member.filename).resolve()
                if not str(target).startswith(str(extract_dir.resolve())):
                    raise ValueError("Unsafe archive path")
            zf.extractall(extract_dir)
        hea = next(extract_dir.rglob("*.hea"), None)
        if hea is None:
            raise ValueError("WFDB ZIP must contain a .hea record header")
        return token, hea.with_suffix("")
    return token, destination


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/sample_data/<path:filename>")
def sample_data(filename: str):
    return send_from_directory(SAMPLE_DATA, filename)


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "No recording uploaded."}), 400
    raw = uploaded.read()
    try:
        df = load_csv(raw)
        result = validate_dataframe(df, request.form.get("time_col") or None)
        payload = {"valid": result.valid, "errors": result.errors, "warnings": result.warnings, "time_col": result.time_col, "signal_cols": result.signal_cols, "sampling_rate_hz": result.sampling_rate_hz, "duration_s": result.duration_s, "n_samples": result.n_samples, "format": "csv", "filename": uploaded.filename or "recording.csv"}
        if result.valid:
            payload["time"] = df[result.time_col].astype(float).tolist()
            payload["signals"] = {c: df[c].astype(float).fillna(0).tolist() for c in result.signal_cols}
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": f"Could not read recording: {exc}"}), 400


@app.post("/api/recording")
def upload_recording():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "No recording uploaded."}), 400
    try:
        token, path = _save_upload(uploaded)
        recording = load_recording(path)
        return jsonify({"recording_id": token, "filename": uploaded.filename, "format": recording.source_format, "sampling_rate_hz": recording.sampling_rate_hz, "duration_s": float(recording.time[-1]) if len(recording.time) else 0.0, "n_samples": len(recording.time), "channels": list(recording.signals.keys())})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/recording/<recording_id>/window")
def recording_window(recording_id: str):
    try:
        matches = list(UPLOAD_ROOT.glob(recording_id)) + list(UPLOAD_ROOT.glob(recording_id + ".*"))
        path = matches[0] if matches else None
        if path is None and (UPLOAD_ROOT / recording_id).is_dir():
            path = next((p.with_suffix("") for p in (UPLOAD_ROOT / recording_id).rglob("*.hea")), None)
        if path is None:
            return jsonify({"error": "recording not found"}), 404
        recording = load_recording(path)
        start = max(0, int(request.args.get("start", 0)))
        stop = min(len(recording.time), int(request.args.get("stop", min(start + 5000, len(recording.time)))))
        if stop <= start:
            raise ValueError("invalid window")
        return jsonify({"start": start, "stop": stop, "time": recording.time[start:stop].tolist(), "signals": {c: y[start:stop].tolist() for c, y in recording.signals.items()}})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/filter")
def filter_signal():
    data = request.get_json(silent=True) or {}
    try:
        fs = float(data["sampling_rate_hz"])
        raw = np.asarray(data["signal"], dtype=float)
        filtered = apply_pipeline(raw, fs, baseline=bool(data.get("baseline", False)), highpass_hz=float(data["highpass_hz"]) if data.get("highpass_hz") is not None else None, lowpass_hz=float(data["lowpass_hz"]) if data.get("lowpass_hz") is not None else None, notch_hz=float(data["notch_hz"]) if data.get("notch_hz") is not None else None)
        return jsonify({"signal": filtered.tolist()})
    except (KeyError, TypeError, ValueError, FilterConfigurationError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/detect/r-peaks")
def detect_r_peaks():
    data = request.get_json(silent=True) or {}
    try:
        fs = float(data["sampling_rate_hz"])
        y = np.asarray(data["signal"], dtype=float)
        distance_ms = float(data.get("min_distance_ms", 250))
        prominence_factor = float(data.get("prominence_factor", 0.5))
        if distance_ms <= 0 or prominence_factor < 0:
            raise ValueError("peak detector settings must be valid positive values")
        distance = max(1, int(round(fs * distance_ms / 1000.0)))
        centered = np.nan_to_num(y - np.nanmedian(y))
        scale = float(np.nanstd(centered)) or 1.0
        peaks, props = sps.find_peaks(centered, distance=distance, prominence=scale * prominence_factor)
        return jsonify({"peaks": peaks.astype(int).tolist(), "prominences": props.get("prominences", np.zeros(len(peaks))).astype(float).tolist()})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/beats")
def beats_endpoint():
    data = request.get_json(silent=True) or {}
    try:
        time = np.asarray(data["time"], dtype=float)
        peaks = np.asarray(data["peaks"], dtype=int)
        beats = segment_beats(time, peaks, float(data.get("pre_s", 0.35)), float(data.get("post_s", 0.55)))
        return jsonify({"beats": [b.to_dict() for b in beats]})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/phenotype")
def phenotype_endpoint():
    data = request.get_json(silent=True) or {}
    try:
        phenotypes = beat_phenotypes(np.asarray(data["time"], dtype=float), np.asarray(data["signal"], dtype=float), np.asarray(data["r_indices"], dtype=int))
        return jsonify({"beats": phenotypes, "summary": summary_statistics(phenotypes)})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/statistics/compare")
def statistics_compare():
    data = request.get_json(silent=True) or {}
    try:
        result = compare_groups(data["group_a"], data["group_b"])
        return jsonify(result)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/statistics/fdr")
def statistics_fdr():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify({"adjusted_p": benjamini_hochberg(data["p_values"])})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/benchmark")
def benchmark_endpoint():
    data = request.get_json(silent=True) or {}
    try:
        result = benchmark_models(np.asarray(data["X"], dtype=float), np.asarray(data["y"]), np.asarray(data["groups"]), int(data.get("folds", 5)))
        return jsonify(result)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/project")
def project_get():
    name = request.args.get("name", "default")
    store = ProjectStore(PROJECT_ROOT / name)
    return jsonify(store.load().to_dict())


@app.post("/api/project/recording")
def project_recording():
    data = request.get_json(silent=True) or {}
    try:
        name = str(data.get("project", "default"))
        store = ProjectStore(PROJECT_ROOT / name)
        ref = RecordingRef(recording_id=str(data["recording_id"]), subject_id=str(data["subject_id"]), group=str(data.get("group", "")), visit=str(data.get("visit", "")), source_path=str(data.get("source_path", "")), format=str(data.get("format", "")), sampling_rate_hz=data.get("sampling_rate_hz"), duration_s=data.get("duration_s"), channels=list(data.get("channels", [])), metadata=dict(data.get("metadata", {})))
        return jsonify(store.add_recording(ref).to_dict())
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/<path:path>")
def static_file(path: str):
    return send_from_directory(WEB, path)


if __name__ == "__main__":
    app.run(host=os.getenv("ELECTROTRACE_HOST", "127.0.0.1"), port=int(os.getenv("ELECTROTRACE_PORT", "5000")), debug=False)
