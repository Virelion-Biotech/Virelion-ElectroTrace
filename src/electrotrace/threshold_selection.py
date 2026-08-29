"""Threshold selection utilities for Stage-2 candidate suppression."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def select_threshold_for_f1(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    min_recall: float = 0.97,
) -> float:
    """Choose probability threshold maximizing F1 subject to recall >= min_recall."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    min_recall = float(min_recall)
    if y_true.shape != probabilities.shape or y_true.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional and equal length")
    if not (0 < min_recall <= 1) or not np.isfinite(min_recall):
        raise ValueError("min_recall must be in (0, 1]")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    positives = int(y_true.sum())
    if positives == 0:
        raise ValueError("at least one positive candidate is required")
    order = np.argsort(-probabilities)
    y_sorted = y_true[order]
    p_sorted = probabilities[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    valid = np.flatnonzero(recall >= min_recall)
    if valid.size == 0:
        return float(p_sorted[-1])
    best = valid[np.argmax(f1[valid])]
    return float(p_sorted[best])
