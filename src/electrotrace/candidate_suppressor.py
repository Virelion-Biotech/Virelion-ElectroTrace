"""Second-stage false-positive suppression for ECG R-peak candidates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
from scipy import signal as sps
from sklearn.ensemble import RandomForestClassifier

SUPPRESSOR_VERSION = "rf-candidate-suppressor-v1"
FEATURE_SCHEMA_VERSION = "candidate-features-v1"
DEFAULT_TARGET_RECALL = 0.995
DEFAULT_TOLERANCE_S = 0.075


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


def _candidate_features(
    signal: np.ndarray,
    fs_hz: float,
    candidate_indices: Sequence[int],
    prominences: Sequence[float] | None = None,
    window_s: float = 0.25,
) -> tuple[np.ndarray, list[str]]:
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
    if prominences is None:
        prominences = np.zeros(len(candidates), dtype=float)
    prominences = np.asarray(prominences, dtype=float)
    if prominences.ndim != 1 or len(prominences) != len(candidates):
        raise ValueError("prominences must match candidate_indices")

    if len(candidates) > 1:
        rr = np.diff(candidates) / fs_hz
        rr_median = float(np.median(rr)) if rr.size else 1.0
    else:
        rr = np.asarray([], dtype=float)
        rr_median = 1.0

    names = [
        "amplitude_z", "prominence_z", "width_s", "max_abs_slope", "mean_abs_slope",
        "local_rms", "crest_factor", "low_band_fraction", "qrs_band_fraction",
        "high_band_fraction", "very_high_band_fraction", "left_right_energy_ratio",
        "left_right_amplitude_ratio", "rr_prev_s", "rr_next_s", "rr_prev_ratio",
        "rr_next_ratio",
    ] + [f"shape_{i:02d}" for i in range(64)]

    rows: list[np.ndarray] = []
    radius = max(16, int(round(fs_hz * window_s)))
    half_qrs = max(4, int(round(fs_hz * 0.08)))

    for pos, candidate in enumerate(candidates):
        lo = max(0, int(candidate) - radius)
        hi = min(signal.size, int(candidate) + radius + 1)
        x = zsignal[lo:hi]
        if x.size < 16:
            x = np.pad(x, (0, 16 - x.size), mode="edge")
        shape = np.interp(np.linspace(0, x.size - 1, 64), np.arange(x.size), x)

        derivative = np.diff(x) * fs_hz
        freqs, power = sps.periodogram(x, fs=fs_hz)
        total_power = float(power.sum()) or 1.0
        bands = [
            float(power[(freqs >= a) & (freqs < b)].sum() / total_power)
            for a, b in ((0.5, 5), (5, 15), (15, 40), (40, min(100.0, fs_hz / 2.0)))
        ]

        qlo = max(lo, int(candidate) - half_qrs)
        qhi = min(hi, int(candidate) + half_qrs + 1)
        qrs = zsignal[qlo:qhi]
        local_rms = float(np.sqrt(np.mean(qrs * qrs))) if qrs.size else 0.0
        peak_abs = float(np.max(np.abs(qrs))) if qrs.size else 0.0
        mean_abs = float(np.mean(np.abs(qrs))) if qrs.size else 0.0
        crest = peak_abs / max(mean_abs, 1e-6)

        left = zsignal[max(lo, int(candidate) - half_qrs):int(candidate)]
        right = zsignal[int(candidate) + 1:min(hi, int(candidate) + half_qrs + 1)]
        left_energy = float(np.mean(left * left)) if left.size else 0.0
        right_energy = float(np.mean(right * right)) if right.size else 0.0
        left_amp = float(np.max(np.abs(left))) if left.size else 0.0
        right_amp = float(np.max(np.abs(right))) if right.size else 0.0

        # Stable local half-amplitude width. We deliberately avoid peak_widths()
        # here because Stage-1 candidates may have zero measured prominence,
        # which otherwise produces noisy PeakPropertyWarnings.
        half_height = 0.5 * peak_abs
        above = np.abs(qrs) >= half_height if qrs.size else np.asarray([], dtype=bool)
        width = float(np.count_nonzero(above) / fs_hz) if above.size and half_height > 0 else 0.0

        prev_rr = float(candidates[pos] - candidates[pos - 1]) / fs_hz if pos else rr_median
        next_rr = float(candidates[pos + 1] - candidates[pos]) / fs_hz if pos + 1 < len(candidates) else rr_median
        rows.append(np.asarray([
            float(zsignal[candidate]),
            float(prominences[pos] / global_scale),
            width,
            float(np.max(np.abs(derivative))) if derivative.size else 0.0,
            float(np.mean(np.abs(derivative))) if derivative.size else 0.0,
            local_rms,
            crest,
            bands[0], bands[1], bands[2], bands[3],
            left_energy / max(right_energy, 1e-6),
            left_amp / max(right_amp, 1e-6),
            prev_rr,
            next_rr,
            prev_rr / max(rr_median, 1e-6),
            next_rr / max(rr_median, 1e-6),
            *shape,
        ], dtype=np.float32))

    if not rows:
        return np.empty((0, len(names)), dtype=np.float32), names
    return np.vstack(rows), names


def label_candidates(
    candidate_indices: Sequence[int],
    reference_indices: Sequence[int],
    fs_hz: float,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> np.ndarray:
    """Label Stage-1 candidates as true/false using one-to-one timing proximity."""
    fs_hz = float(fs_hz)
    tolerance_s = float(tolerance_s)
    candidates = np.asarray(candidate_indices, dtype=int)
    references = np.asarray(reference_indices, dtype=int)
    if fs_hz <= 0 or not np.isfinite(fs_hz):
        raise ValueError("fs_hz must be positive and finite")
    if tolerance_s <= 0 or not np.isfinite(tolerance_s):
        raise ValueError("tolerance_s must be positive and finite")
    if candidates.ndim != 1 or references.ndim != 1:
        raise ValueError("candidate_indices and reference_indices must be one-dimensional")
    if not np.all(candidates[1:] >= candidates[:-1]) or not np.all(references[1:] >= references[:-1]):
        raise ValueError("candidate_indices and reference_indices must be sorted")

    labels = np.zeros(len(candidates), dtype=np.int8)
    used_reference = np.zeros(len(references), dtype=bool)
    tolerance_samples = tolerance_s * fs_hz
    for i, candidate in enumerate(candidates):
        pos = int(np.searchsorted(references, candidate))
        possible = []
        if pos < len(references):
            possible.append(pos)
        if pos > 0:
            possible.append(pos - 1)
        if not possible:
            continue
        best = min(possible, key=lambda j: abs(int(references[j]) - int(candidate)))
        if not used_reference[best] and abs(int(references[best]) - int(candidate)) <= tolerance_samples:
            labels[i] = 1
            used_reference[best] = True
    return labels


def select_threshold_for_recall(y_true: Sequence[int], probabilities: Sequence[float], target_recall: float = DEFAULT_TARGET_RECALL) -> float:
    """Select the highest probability threshold that achieves target recall on training data."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    target_recall = float(target_recall)
    if y_true.shape != probabilities.shape or y_true.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional and equal length")
    if not (0 < target_recall <= 1) or not np.isfinite(target_recall):
        raise ValueError("target_recall must be in (0, 1]")
    positives = int(y_true.sum())
    if positives == 0:
        raise ValueError("at least one positive candidate is required")
    order = np.argsort(-probabilities)
    ranked_labels = y_true[order]
    cumulative_tp = np.cumsum(ranked_labels)
    recall = cumulative_tp / positives
    valid = np.flatnonzero(recall >= target_recall)
    if valid.size == 0:
        return 0.0
    return float(probabilities[order[valid[0]]])


class CandidateSuppressor:
    """Random-forest verifier for high-recall Stage-1 ECG candidates."""

    def __init__(self, model: RandomForestClassifier | None = None, metadata: SuppressorMetadata | None = None):
        self.model = model
        self.metadata = metadata
        self.feature_names: list[str] | None = None

    @property
    def fitted(self) -> bool:
        return self.model is not None and self.metadata is not None

    def fit(
        self,
        feature_matrix: np.ndarray,
        labels: Sequence[int],
        *,
        target_recall: float = DEFAULT_TARGET_RECALL,
        random_seed: int = 42,
        n_estimators: int = 300,
    ) -> "CandidateSuppressor":
        X = np.asarray(feature_matrix, dtype=float)
        y = np.asarray(labels, dtype=int)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("feature_matrix must be a non-empty two-dimensional array")
        if len(y) != len(X):
            raise ValueError("feature_matrix and labels must have equal length")
        if not np.isfinite(X).all():
            raise ValueError("feature_matrix contains NaN or infinite values")
        if set(np.unique(y)) - {0, 1} or len(np.unique(y)) < 2:
            raise ValueError("labels must contain both negative and positive candidates")
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_depth=14,
            random_state=int(random_seed),
            n_jobs=-1,
        )
        model.fit(X, y)
        probabilities = model.predict_proba(X)[:, 1]
        threshold = select_threshold_for_recall(y, probabilities, target_recall=target_recall)
        self.model = model
        self.metadata = SuppressorMetadata(
            model_version=SUPPRESSOR_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            target_recall=float(target_recall),
            threshold=float(threshold),
            n_training_candidates=int(len(y)),
            n_positive_candidates=int(y.sum()),
            n_negative_candidates=int((y == 0).sum()),
            random_seed=int(random_seed),
            n_estimators=int(n_estimators),
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("candidate suppressor is not fitted")
        X = np.asarray(features, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("features must be a finite two-dimensional array")
        return self.model.predict_proba(X)[:, 1]

    def filter_candidates(self, candidate_indices: Sequence[int], features: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(features)
        if len(candidate_indices) != len(probabilities):
            raise ValueError("candidate_indices and features must have equal length")
        selected_threshold = float(self.metadata.threshold if threshold is None else threshold)
        if not 0 <= selected_threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        candidates = np.asarray(candidate_indices, dtype=int)
        mask = probabilities >= selected_threshold
        return candidates[mask], probabilities

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise ValueError("cannot save an unfitted candidate suppressor")
        payload = {
            "model": self.model,
            "metadata": self.metadata,
            "feature_names": self.feature_names,
        }
        Path(path).write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str | Path) -> "CandidateSuppressor":
        payload = pickle.loads(Path(path).read_bytes())
        obj = cls(model=payload["model"], metadata=payload["metadata"])
        obj.feature_names = payload.get("feature_names")
        return obj
