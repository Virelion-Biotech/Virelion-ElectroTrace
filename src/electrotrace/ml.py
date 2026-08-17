"""Lightweight model-assisted beat classification and uncertainty sampling."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _validate_signal(signal: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    signal = np.asarray(signal, dtype=float)
    fs = float(fs)
    if signal.ndim != 1 or len(signal) < 8:
        raise ValueError("signal must be one-dimensional with at least eight samples")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling rate must be positive and finite")
    if not np.isfinite(signal).any():
        raise ValueError("signal contains no finite values")
    return signal, fs


def _beat_features(signal: np.ndarray, fs: float, center_index: int, window_s: float = 0.8) -> np.ndarray:
    signal, fs = _validate_signal(signal, fs)
    if center_index < 0 or center_index >= len(signal):
        raise ValueError("beat index is outside the signal")
    radius = max(8, int(round(fs * window_s / 2)))
    lo, hi = max(0, center_index - radius), min(len(signal), center_index + radius + 1)
    x = np.asarray(signal[lo:hi], dtype=float)
    if x.size < 8:
        x = np.pad(x, (0, max(0, 8 - x.size)), mode="edge")
    x = np.nan_to_num(x)
    med = np.median(x)
    scale = np.std(x) or 1.0
    z = (x - med) / scale
    d = np.diff(z)
    freqs, power = sps.periodogram(z, fs=fs)
    bands = [float(power[(freqs >= a) & (freqs < b)].sum()) for a, b in ((0.5, 5), (5, 15), (15, 40), (40, min(100, fs / 2)))]
    return np.asarray([
        float(np.mean(z)), float(np.std(z)), float(np.max(z)), float(np.min(z)),
        float(np.max(np.abs(z))), float(np.mean(np.abs(d))), *bands,
    ], dtype=float)


def _times_from_indices(beat_indices: np.ndarray, fs: float, time: np.ndarray | None) -> np.ndarray:
    beat_indices = np.asarray(beat_indices, dtype=int)
    if np.any(beat_indices < 0):
        raise ValueError("beat indices cannot be negative")
    if time is None:
        return beat_indices.astype(float) / fs
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) == 0 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    if np.any(beat_indices >= len(time)):
        raise ValueError("beat index exceeds time-axis length")
    return time[beat_indices]


def build_training_set(signal: np.ndarray, fs: float, beat_indices: np.ndarray, annotations: list[dict], tolerance_s: float = 0.08, time: np.ndarray | None = None):
    signal, fs = _validate_signal(signal, fs)
    beat_indices = np.asarray(beat_indices, dtype=int)
    if beat_indices.size == 0:
        raise ValueError("Need detected beats before training")
    if tolerance_s <= 0 or not np.isfinite(tolerance_s):
        raise ValueError("tolerance_s must be positive and finite")
    points = [a for a in annotations if a.get("type") == "point" and a.get("status") == "accepted" and a.get("time") is not None and a.get("label")]
    if not points:
        raise ValueError("Need accepted point annotations before training")
    beat_times = _times_from_indices(beat_indices, fs, time)
    X, y = [], []
    for idx, t in zip(beat_indices, beat_times):
        candidates = [(abs(float(p["time"]) - float(t)), p) for p in points]
        distance, point = min(candidates, key=lambda pair: pair[0])
        if distance <= tolerance_s:
            X.append(_beat_features(signal, fs, int(idx)))
            y.append(str(point["label"]))
    if not X:
        raise ValueError("No detected beats fall within the annotation matching tolerance")
    if len(set(y)) < 2:
        raise ValueError("Need accepted annotations from at least two labels for supervised training")
    return np.vstack(X), np.asarray(y)


def train_classifier(signal: np.ndarray, fs: float, beat_indices: np.ndarray, annotations: list[dict], time: np.ndarray | None = None):
    X, y = build_training_set(signal, fs, beat_indices, annotations, time=time)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    model.fit(X, y)
    return model, {"n_training_examples": int(len(y)), "classes": sorted(set(map(str, y)))}


def rank_uncertain(signal: np.ndarray, fs: float, beat_indices: np.ndarray, model, top_n: int = 25, time: np.ndarray | None = None) -> list[dict]:
    signal, fs = _validate_signal(signal, fs)
    beat_indices = np.asarray(beat_indices, dtype=int)
    if beat_indices.size == 0:
        return []
    top_n = int(top_n)
    if top_n <= 0:
        return []
    features = np.vstack([_beat_features(signal, fs, int(i)) for i in beat_indices])
    probabilities = model.predict_proba(features)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-9, 1.0))).sum(axis=1)
    times = _times_from_indices(beat_indices, fs, time)
    order = np.argsort(-entropy)[: min(top_n, len(beat_indices))]
    return [
        {
            "index": int(beat_indices[i]),
            "time_s": float(times[i]),
            "predicted_label": str(predictions[i]),
            "confidence": float(np.max(probabilities[i])),
            "uncertainty": float(entropy[i]),
        }
        for i in order
    ]
