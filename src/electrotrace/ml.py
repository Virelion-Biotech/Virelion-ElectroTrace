"""Lightweight model-assisted beat classification and uncertainty sampling."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _beat_features(signal: np.ndarray, fs: float, center_index: int, window_s: float = 0.8) -> np.ndarray:
    n = len(signal)
    radius = max(8, int(round(fs * window_s / 2)))
    lo, hi = max(0, center_index - radius), min(n, center_index + radius + 1)
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


def build_training_set(signal: np.ndarray, fs: float, beat_indices: np.ndarray, annotations: list[dict], tolerance_s: float = 0.08):
    points = [a for a in annotations if a.get("type") == "point" and a.get("status") == "accepted" and a.get("time") is not None]
    if not points:
        raise ValueError("Need accepted point annotations before training")
    X, y = [], []
    beat_times = beat_indices / fs
    for idx, t in zip(beat_indices, beat_times):
        distances = [abs(float(p["time"]) - float(t)) for p in points]
        if not distances:
            continue
        j = int(np.argmin(distances))
        if distances[j] <= tolerance_s:
            X.append(_beat_features(signal, fs, int(idx)))
            y.append(points[j]["label"])
    if len(set(y)) < 2:
        raise ValueError("Need accepted annotations from at least two labels for supervised training")
    return np.vstack(X), np.asarray(y)


def train_classifier(signal: np.ndarray, fs: float, beat_indices: np.ndarray, annotations: list[dict]):
    X, y = build_training_set(signal, fs, beat_indices, annotations)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    model.fit(X, y)
    return model, {"n_training_examples": int(len(y)), "classes": sorted(set(map(str, y)))}


def rank_uncertain(signal: np.ndarray, fs: float, beat_indices: np.ndarray, model, top_n: int = 25) -> list[dict]:
    features = np.vstack([_beat_features(signal, fs, int(i)) for i in beat_indices])
    probabilities = model.predict_proba(features)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-9, 1.0))).sum(axis=1)
    order = np.argsort(-entropy)[: max(0, min(top_n, len(beat_indices)))]
    return [
        {
            "index": int(beat_indices[i]),
            "time_s": float(beat_indices[i] / fs),
            "predicted_label": str(predictions[i]),
            "confidence": float(np.max(probabilities[i])),
            "uncertainty": float(entropy[i]),
        }
        for i in order
    ]
