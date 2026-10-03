"""Label-free per-lead quality summaries for multi-lead selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .candidate_suppressor import CandidateSuppressor, _candidate_features
from .scale_estimation import estimate_stage1_scale
from .validation_detectors import (
    _candidate_set,
    detect_r_peaks_two_stage,
    select_signal_polarity,
)


@dataclass(frozen=True)
class LeadQuality:
    retained_probability_p50: float
    retained_qrs_band_fraction: float
    retention_fraction: float
    stage1_candidate_count: int
    retained_count: int
    selected_polarity: str
    polarity_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_lead_quality(
    signal: np.ndarray,
    fs_hz: float,
    suppressor: CandidateSuppressor,
    *,
    scale_method: str,
    v2_gate_confidence: float,
    width_override_confidence: float,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, LeadQuality]:
    """Run the frozen detector and derive label-free selector inputs.

    No reference annotation is accepted by this API. The Stage-1 candidate
    stream is reconstructed using the same adaptive-polarity decision and
    scale method as the canonical detector. Final retained samples must be a
    subset of that stream; otherwise the function fails rather than deriving
    mismatched quality features.
    """
    x = np.asarray(signal, dtype=float)
    fs_hz = float(fs_hz)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("signal must be non-empty, one-dimensional, and finite")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be positive and finite")
    if not suppressor.fitted:
        raise ValueError("suppressor must be fitted")

    decision = select_signal_polarity(
        x,
        fs_hz,
        scale_method=scale_method,
        v2_gate_confidence=v2_gate_confidence,
        width_override_confidence=width_override_confidence,
    )
    centered = x - np.median(x)
    scale = estimate_stage1_scale(centered, fs_hz, method=scale_method)
    stream = centered if decision.polarity != "negative" else -centered
    candidates, prominences = _candidate_set(stream, fs_hz, scale)

    if candidates.size:
        features, names = _candidate_features(
            x,
            fs_hz,
            candidates,
            prominences,
            scale_method=scale_method,
        )
    else:
        features = np.empty((0, 0), dtype=float)
        names = []

    retained, retained_probabilities = detect_r_peaks_two_stage(
        x,
        fs_hz,
        suppressor,
        threshold=threshold,
        polarity="adaptive",
        recovery=False,
        scale_method=scale_method,
        width_override_confidence=width_override_confidence,
        v2_gate_confidence=v2_gate_confidence,
    )
    retained = np.asarray(retained, dtype=int)
    retained_probabilities = np.asarray(retained_probabilities, dtype=float)
    if retained.ndim != 1 or retained_probabilities.ndim != 1:
        raise RuntimeError("canonical detector returned invalid retained arrays")
    if retained.size != retained_probabilities.size:
        raise RuntimeError("retained samples and probabilities differ in length")
    if not np.isfinite(retained_probabilities).all():
        raise RuntimeError("canonical detector returned non-finite probabilities")
    if retained.size and not np.all(np.isin(retained, candidates)):
        raise RuntimeError(
            "canonical retained samples are absent from reconstructed Stage-1 candidates"
        )

    retained_mask = np.isin(candidates, retained)
    if retained.size:
        if not names or "qrs_band_fraction" not in names:
            raise RuntimeError("candidate feature schema lacks qrs_band_fraction")
        qrs_index = names.index("qrs_band_fraction")
        qrs_values = features[retained_mask, qrs_index]
        if qrs_values.size != retained.size or not np.isfinite(qrs_values).all():
            raise RuntimeError("invalid retained QRS-band feature values")
        qrs_median = float(np.median(qrs_values))
        p50 = float(np.median(retained_probabilities))
    else:
        qrs_median = 0.0
        p50 = 0.0

    retention = float(retained.size / candidates.size) if candidates.size else 0.0
    for name, value in (
        ("retained_probability_p50", p50),
        ("retained_qrs_band_fraction", qrs_median),
        ("retention_fraction", retention),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{name} is outside [0, 1]: {value}")

    quality = LeadQuality(
        retained_probability_p50=p50,
        retained_qrs_band_fraction=qrs_median,
        retention_fraction=retention,
        stage1_candidate_count=int(candidates.size),
        retained_count=int(retained.size),
        selected_polarity=str(decision.polarity),
        polarity_confidence=float(decision.confidence),
    )
    return retained, retained_probabilities, quality
