"""Small, transparent statistical analysis helpers for phenotype tables."""
from __future__ import annotations

import math
import numpy as np
from scipy import stats


def compare_groups(values_a: list[float], values_b: list[float]) -> dict:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least two finite observations")
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled = math.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / (len(a)+len(b)-2))
    effect = float((np.mean(a)-np.mean(b))/pooled) if pooled > 0 else 0.0
    return {
        "n_a": int(len(a)), "n_b": int(len(b)),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "welch_t": float(t.statistic), "welch_p": float(t.pvalue),
        "mann_whitney_u": float(u.statistic), "mann_whitney_p": float(u.pvalue),
        "cohens_d": effect,
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()
