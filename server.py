"""ElectroTrace local web server. Run with: python server.py"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from scipy import signal as sps

from electrotrace.io import load_csv, validate_dataframe
from electrotrace.signal import apply_pipeline, FilterConfigurationError

WEB = os.path.join(ROOT, "web")
SAMPLE_DATA = os.path.join(ROOT, "sample_data")
app = Flask(__name__, static_folder=WEB, static_url_path="")

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
        return jsonify({"error": "No CSV file uploaded."}), 400
    raw = uploaded.read()
    try:
        df = load_csv(raw)
        selected_time = request.form.get("time_col") or None
        result = validate_dataframe(df, selected_time)
        payload = {
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "time_col": result.time_col,
            "signal_cols": result.signal_cols,
            "sampling_rate_hz": result.sampling_rate_hz,
            "duration_s": result.duration_s,
            "n_samples": result.n_samples,
            "filename": uploaded.filename or "recording.csv",
        }
        if result.valid:
            payload["time"] = df[result.time_col].astype(float).tolist()
            payload["signals"] = {c: df[c].astype(float).fillna(0).tolist() for c in result.signal_cols}
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": f"Could not read recording: {exc}"}), 400

@app.post("/api/filter")
def filter_signal():
    data = request.get_json(silent=True) or {}
    try:
        fs = float(data["sampling_rate_hz"])
        raw = np.asarray(data["signal"], dtype=float)
        filtered = apply_pipeline(
            raw,
            fs,
            baseline=bool(data.get("baseline", False)),
            highpass_hz=float(data["highpass_hz"]) if data.get("highpass_hz") is not None else None,
            lowpass_hz=float(data["lowpass_hz"]) if data.get("lowpass_hz") is not None else None,
            notch_hz=float(data["notch_hz"]) if data.get("notch_hz") is not None else None,
        )
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
        return jsonify({
            "peaks": peaks.astype(int).tolist(),
            "prominences": props.get("prominences", np.zeros(len(peaks))).astype(float).tolist(),
        })
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.get("/<path:path>")
def static_file(path: str):
    return send_from_directory(WEB, path)

if __name__ == "__main__":
    host = os.getenv("ELECTROTRACE_HOST", "127.0.0.1")
    port = int(os.getenv("ELECTROTRACE_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
