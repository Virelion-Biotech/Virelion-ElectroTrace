"""Second-stage false-positive suppression for ECG R-peak candidates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
from scipy import signal as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

SUPPRESSOR_VERSION = "rf-candidate-suppressor-v4"
FEATURE_SCHEMA_VERSION = "candidate-features-v3"
DEFAULT_TARGET_RECALL = 0.995
DEFAULT_TOLERANCE_S = 0.075
DEFAULT_CALIBRATION_FRACTION = 0.20
FEATURE_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class SuppressorMetadata:
    model_version: str
    feature_schema_version: str
    target_recall: float
    threshold: float
    n_training_candidates: int
    n_positive_candidates: int
    n_negative_candidates: int
    random_seed: int
    n_estimators: int
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION
    calibration_candidates: int = 0
    calibration_method: str = "held_out_stratified"

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_signal(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, float]:
    signal = np.asarray(signal, dtype=float)
    fs_hz = float(fs_hz)
    if signal.ndim != 1 or signal.size < 16:
        raise ValueError("signal must be one-dimensional with at least 16 samples")
    if not np.isfinite(signal).all():
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be positive and finite")
    return signal, fs_hz


def _bandpasses(zsignal: np.ndarray, fs_hz: float) -> list[np.ndarray]:
    nyq = fs_hz / 2.0
    specs = [(0.5, min(5.0, nyq * 0.9)), (5.0, min(15.0, nyq * 0.9)),
             (15.0, min(40.0, nyq * 0.9)), (40.0, min(100.0, nyq * 0.9))]
    out: list[np.ndarray] = []
    for low, high in specs:
        if high <= low or high <= 0:
            out.append(np.zeros_like(zsignal))
            continue
        sos = sps.butter(3, [low, high], btype="bandpass", fs=fs_hz, output="sos")
        try:
            out.append(sps.sosfiltfilt(sos, zsignal))
        except ValueError:
            out.append(np.zeros_like(zsignal))
    return out


def _candidate_features(
    signal: np.ndarray,
    fs_hz: float,
    candidate_indices: Sequence[int],
    prominences: Sequence[float] | None = None,
    window_s: float = 0.25,
) -> tuple[np.ndarray, list[str]]:
    """Extract candidate features in bounded vectorized chunks."""
    signal, fs_hz = _validate_signal(signal, fs_hz)
    candidates = np.asarray(candidate_indices, dtype=int)
    if candidates.ndim != 1:
        raise ValueError("candidate_indices must be one-dimensional")
    if np.any(candidates < 0) or np.any(candidates >= signal.size):
        raise ValueError("candidate_indices contain out-of-range values")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be positive and finite")

    base = signal - np.median(signal)
    global_scale = float(np.std(base)) or 1.0
    zsignal = base / global_scale
    prominences = np.zeros(len(candidates), dtype=float) if prominences is None else np.asarray(prominences, dtype=float)
    if prominences.ndim != 1 or len(prominences) != len(candidates):
        raise ValueError("prominences must match candidate_indices")
    if not np.isfinite(prominences).all():
        raise ValueError("prominences must be finite")

    rr = np.diff(candidates) / fs_hz if len(candidates) > 1 else np.asarray([], dtype=float)
    rr_median = float(np.median(rr)) if rr.size else 1.0
    band_signals = _bandpasses(zsignal, fs_hz)

    names = [
        "amplitude_z", "prominence_z", "width_s", "max_abs_slope", "mean_abs_slope",
        "local_rms", "crest_factor", "low_band_fraction", "qrs_band_fraction",
        "high_band_fraction", "very_high_band_fraction", "left_right_energy_ratio",
        "left_right_amplitude_ratio", "rr_prev_s", "rr_next_s", "rr_prev_ratio",
        "rr_next_ratio",
    ] + [f"shape_{i:02d}" for i in range(64)]

    radius = max(16, int(round(fs_hz * window_s)))
    half_qrs = max(4, int(round(fs_hz * 0.08)))
    offsets = np.arange(-radius, radius + 1, dtype=int)
    qslice = slice(radius - half_qrs, radius + half_qrs + 1)
    leftslice = slice(radius - half_qrs, radius)
    rightslice = slice(radius + 1, radius + half_qrs + 1)
    shape_pos = np.linspace(0.0, len(offsets) - 1.0, 64)
    shape_lo = np.floor(shape_pos).astype(int)
    shape_hi = np.minimum(shape_lo + 1, len(offsets) - 1)
    shape_alpha = (shape_pos - shape_lo).astype(np.float32)

    rows: list[np.ndarray] = []
    n = len(candidates)
    for start in range(0, n, FEATURE_CHUNK_SIZE):
        stop = min(n, start + FEATURE_CHUNK_SIZE)
        cand = candidates[start:stop]
        idx = np.clip(cand[:, None] + offsets[None, :], 0, signal.size - 1)
        x = zsignal[idx]
        shape = x[:, shape_lo] * (1.0 - shape_alpha[None, :]) + x[:, shape_hi] * shape_alpha[None, :]
        derivative = np.diff(x, axis=1) * fs_hz
        qrs = x[:, qslice]
        local_rms = np.sqrt(np.mean(qrs * qrs, axis=1))
        peak_abs = np.max(np.abs(qrs), axis=1)
        mean_abs = np.mean(np.abs(qrs), axis=1)
        crest = peak_abs / np.maximum(mean_abs, 1e-6)
        width = np.count_nonzero(np.abs(qrs) >= (0.5 * peak_abs)[:, None], axis=1) / fs_hz

        left = x[:, leftslice]
        right = x[:, rightslice]
        left_energy = np.mean(left * left, axis=1)
        right_energy = np.mean(right * right, axis=1)
        left_amp = np.max(np.abs(left), axis=1)
        right_amp = np.max(np.abs(right), axis=1)

        fractions = []
        bandq = [band[idx[:, qslice]] for band in band_signals]
        for b in bandq:
            fractions.append(np.mean(b * b, axis=1))
        total_band = np.sum(fractions, axis=0)
        total_band = np.maximum(total_band, 1e-8)
        bands = [v / total_band for v in fractions]

        prev_rr = np.empty(stop - start, dtype=float)
        next_rr = np.empty(stop - start, dtype=float)
        if stop - start:
            prev_rr[0] = rr_median
            if stop - start > 1:
                prev_rr[1:] = np.diff(cand) / fs_hz
            next_rr[:-1] = np.diff(cand) / fs_hz
            next_rr[-1] = rr_median
        prev_ratio = prev_rr / max(rr_median, 1e-6)
        next_ratio = next_rr / max(rr_median, 1e-6)

        chunk = np.column_stack([
            zsignal[cand], prominences[start:stop] / global_scale, width,
            np.max(np.abs(derivative), axis=1), np.mean(np.abs(derivative), axis=1),
            local_rms, crest, *bands,
            left_energy / np.maximum(right_energy, 1e-6),
            left_amp / np.maximum(right_amp, 1e-6),
            prev_rr, next_rr, prev_ratio, next_ratio,
            shape,
        ]).astype(np.float32)
        rows.append(chunk)

    if not rows:
        return np.empty((0, len(names)), dtype=np.float32), names
    out = np.vstack(rows)
    if not np.isfinite(out).all():
        raise ValueError("candidate feature extraction produced NaN or infinite values")
    return out, names


def label_candidates(candidate_indices: Sequence[int], reference_indices: Sequence[int], fs_hz: float,
                     tolerance_s: float = DEFAULT_TOLERANCE_S) -> np.ndarray:
    fs_hz = float(fs_hz); tolerance_s = float(tolerance_s)
    candidates = np.asarray(candidate_indices, dtype=int); references = np.asarray(reference_indices, dtype=int)
    if fs_hz <= 0 or not np.isfinite(fs_hz): raise ValueError("fs_hz must be positive and finite")
    if tolerance_s <= 0 or not np.isfinite(tolerance_s): raise ValueError("tolerance_s must be positive and finite")
    if candidates.ndim != 1 or references.ndim != 1: raise ValueError("candidate_indices and reference_indices must be one-dimensional")
    if not np.all(candidates[1:] >= candidates[:-1]) or not np.all(references[1:] >= references[:-1]):
        raise ValueError("candidate_indices and reference_indices must be sorted")
    labels = np.zeros(len(candidates), dtype=np.int8); used_reference = np.zeros(len(references), dtype=bool)
    tolerance_samples = tolerance_s * fs_hz
    for i, candidate in enumerate(candidates):
        pos = int(np.searchsorted(references, candidate)); possible = []
        if pos < len(references): possible.append(pos)
        if pos > 0: possible.append(pos - 1)
        if possible:
            best = min(possible, key=lambda j: abs(int(references[j]) - int(candidate)))
            if not used_reference[best] and abs(int(references[best]) - int(candidate)) <= tolerance_samples:
                labels[i] = 1; used_reference[best] = True
    return labels


def select_threshold_for_recall(y_true: Sequence[int], probabilities: Sequence[float], target_recall: float = DEFAULT_TARGET_RECALL) -> float:
    y_true = np.asarray(y_true, dtype=int); probabilities = np.asarray(probabilities, dtype=float); target_recall = float(target_recall)
    if y_true.shape != probabilities.shape or y_true.ndim != 1: raise ValueError("y_true and probabilities must be one-dimensional and equal length")
    if not (0 < target_recall <= 1) or not np.isfinite(target_recall): raise ValueError("target_recall must be in (0, 1]")
    if not np.isfinite(probabilities).all(): raise ValueError("probabilities must be finite")
    positives = int(y_true.sum())
    if positives == 0: raise ValueError("at least one positive candidate is required")
    order = np.argsort(-probabilities); recall = np.cumsum(y_true[order]) / positives
    valid = np.flatnonzero(recall >= target_recall)
    return float(probabilities[order[valid[0]]]) if valid.size else 0.0


class CandidateSuppressor:
    def __init__(self, model: RandomForestClassifier | None = None, metadata: SuppressorMetadata | None = None):
        self.model = model; self.metadata = metadata; self.feature_names: list[str] | None = None

    @property
    def fitted(self) -> bool:
        return self.model is not None and self.metadata is not None

    def fit(self, feature_matrix: np.ndarray, labels: Sequence[int], *, target_recall: float = DEFAULT_TARGET_RECALL,
            random_seed: int = 42, n_estimators: int = 150, calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION) -> "CandidateSuppressor":
        X = np.asarray(feature_matrix, dtype=float); y = np.asarray(labels, dtype=int)
        if X.ndim != 2 or X.shape[0] == 0: raise ValueError("feature_matrix must be a non-empty two-dimensional array")
        if len(y) != len(X): raise ValueError("feature_matrix and labels must have equal length")
        if not np.isfinite(X).all(): raise ValueError("feature_matrix contains NaN or infinite values")
        if set(np.unique(y)) - {0, 1} or len(np.unique(y)) < 2: raise ValueError("labels must contain both negative and positive candidates")
        calibration_fraction = float(calibration_fraction)
        if not 0 <= calibration_fraction < 0.5 or not np.isfinite(calibration_fraction):
            raise ValueError("calibration_fraction must be finite and in [0, 0.5)")
        unique, counts = np.unique(y, return_counts=True)
        if calibration_fraction > 0 and (len(y) < 10 or np.min(counts) < 2):
            raise ValueError("held-out threshold calibration requires at least 10 candidates and two examples per class; set calibration_fraction=0 to disable calibration")

        calibration_candidates = 0
        calibration_method = "disabled"
        if calibration_fraction > 0:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=calibration_fraction, random_state=int(random_seed))
            model_idx, calibration_idx = next(splitter.split(X, y))
            calibration_model = RandomForestClassifier(
                n_estimators=int(n_estimators), class_weight="balanced_subsample", min_samples_leaf=2,
                max_depth=14, random_state=int(random_seed), n_jobs=-1,
            )
            calibration_model.fit(X[model_idx], y[model_idx])
            calibration_probabilities = calibration_model.predict_proba(X[calibration_idx])[:, 1]
            threshold = select_threshold_for_recall(y[calibration_idx], calibration_probabilities, target_recall=target_recall)
            calibration_candidates = int(len(calibration_idx))
            calibration_method = "held_out_stratified"
        else:
            final_for_threshold = RandomForestClassifier(
                n_estimators=int(n_estimators), class_weight="balanced_subsample", min_samples_leaf=2,
                max_depth=14, random_state=int(random_seed), n_jobs=-1,
            )
            final_for_threshold.fit(X, y)
            threshold = select_threshold_for_recall(y, final_for_threshold.predict_proba(X)[:, 1], target_recall=target_recall)
            calibration_method = "training_resubstitution"

        model = RandomForestClassifier(n_estimators=int(n_estimators), class_weight="balanced_subsample", min_samples_leaf=2,
                                       max_depth=14, random_state=int(random_seed), n_jobs=-1)
        model.fit(X, y)
        self.model = model
        self.metadata = SuppressorMetadata(SUPPRESSOR_VERSION, FEATURE_SCHEMA_VERSION, float(target_recall), float(threshold),
                                           int(len(y)), int(y.sum()), int((y == 0).sum()), int(random_seed), int(n_estimators),
                                           calibration_fraction=float(calibration_fraction), calibration_candidates=calibration_candidates,
                                           calibration_method=calibration_method)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("candidate suppressor is not fitted")
        X = np.asarray(features, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all(): raise ValueError("features must be a finite two-dimensional array")
        return self.model.predict_proba(X)[:, 1]

    def filter_candidates(self, candidate_indices: Sequence[int], features: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(features)
        if len(candidate_indices) != len(probabilities): raise ValueError("candidate_indices and features must have equal length")
        selected_threshold = float(self.metadata.threshold if threshold is None else threshold)
        if not 0 <= selected_threshold <= 1: raise ValueError("threshold must be in [0, 1]")
        candidates = np.asarray(candidate_indices, dtype=int); mask = probabilities >= selected_threshold
        return candidates[mask], probabilities

    def save(self, path: str | Path) -> None:
        if not self.fitted: raise ValueError("cannot save an unfitted candidate suppressor")
        Path(path).write_bytes(pickle.dumps({"model": self.model, "metadata": self.metadata, "feature_names": self.feature_names}, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str | Path) -> "CandidateSuppressor":
        """Load a suppressor from a trusted local model file.

        Pickle is executable serialization; never load model files obtained from
        untrusted users, downloads, or external sources.
        """
        payload = pickle.loads(Path(path).read_bytes()); obj = cls(model=payload["model"], metadata=payload["metadata"]); obj.feature_names = payload.get("feature_names"); return obj
