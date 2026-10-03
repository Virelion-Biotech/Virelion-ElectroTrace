"""Shared amplitude-scale estimation, used by both Stage 1 (candidate
generation, in validation_detectors.py) and Stage 2 (RF feature
normalization, in candidate_suppressor.py).

Extracted into its own module (rather than living in validation_detectors.py,
where it originated) specifically to avoid a circular import:
validation_detectors.py already imports from candidate_suppressor.py, so
candidate_suppressor.py cannot import back from validation_detectors.py.

--- History ---

Priority-1 fix (see docs/PRIORITY1_STAGE1_HANDOFF.md): Stage-1 candidate
generation used a single global `std(signal)` to set the prominence
threshold for the *entire* record. On INCART, a short high-amplitude
noise/artifact segment inflates that one number, which raises the
threshold everywhere else in the record and starves candidate generation
even over otherwise clean QRS complexes.

Real-data validation on full INCART + MIT-BIH held-out (2026-09-12), four
candidates compared:
  std (original):  INCART F1=0.4854  MIT-BIH F1=0.9651  (baseline)
  mad:              INCART F1=0.3797  -- worse, dropped
  windowed_mad:     INCART F1=0.3358  MIT-BIH F1=0.9691  -- worse on
    INCART: MAD's median-based robustness breaks down once a window's
    "elevated" (QRS/T) fraction exceeds ~50% (faster/wider-complex
    records), flipping the estimate from tracking baseline noise to
    tracking QRS/T amplitude and over-suppressing real beats. Dropped.
  adaptive:         INCART F1=0.3862  MIT-BIH F1=0.9647  -- worse than
    both std and windowed_std individually. Root cause: the stage-2 RF
    is trained on whichever scale_method generated its training
    candidates. adaptive's per-record std/windowed_std switching almost
    never triggers on MIT-BIH (so the RF only ever learns std-shaped
    candidates) but DOES trigger on the hard INCART records at
    inference -- a train/test mismatch. Dropped.
  windowed_std:     INCART F1=0.4716  MIT-BIH F1=0.9651  -- ADOPTED.
    Median of per-window std (not MAD) keeps std's continuous behavior
    (no discrete flip at the ~50% occupancy point) while still
    discounting a minority of artifact-corrupted windows. Zero MIT-BIH
    cost; substantially improves the Priority-1 focus records (I56
    sens 0.0018->0.2628, I24 0.0019->0.0817, I40 0.0079->0.0893, I19
    0.0078->0.0281); costs ~1.4 points of INCART aggregate F1 because
    ~12 mid-performing records get moderately worse even as the worst
    ones improve a lot.

Stage-2 finding (2026-09-13): I03 remained broken even after the Stage-1
fix above, despite Stage-1 now generating 2x the reference beat count in
candidates for I03 (ratio 2.05). Root cause: I03 has ~8mV of near-linear
DC drift across its 30-minute recording (~20x the actual QRS amplitude of
~0.3-0.5mV) -- likely an electrode/contact issue specific to that
recording, not a detector bug. candidate_suppressor.py's
`_candidate_features` computed its own independent
`global_scale = std(signal - median(signal))` for feature normalization,
which has the *exact same* fragile-global-statistic problem as the
original Stage-1 bug: for I03, global_scale ~= 2.39 (dominated by drift),
so true-beat candidates with raw prominence ~0.3-0.5 normalize to ~0.15-0.2
-- indistinguishable from noise to an RF trained mostly on records without
this drift. This is why Stage 1 could generate abundant true candidates
for I03 while Stage 2 still suppressed nearly all of them. Fixed by using
the same estimate_scale() function (windowed_std) for Stage-2 feature
normalization as for Stage-1 candidate generation -- requires a full
MIT-BIH retrain since it changes the feature space the RF is fit on.
"""
from __future__ import annotations

import numpy as np

DEFAULT_SCALE_METHOD = "windowed_std"
DEFAULT_SCALE_WINDOW_S = 8.0
DEFAULT_ADAPTIVE_INFLATION_RATIO = 1.5
_MAD_TO_STD = 1.4826


def _mad_scale(x: np.ndarray) -> float:
    """Robust scale estimate: 1.4826 * MAD, falling back to std if MAD is ~0
    (e.g. a flat/quantized segment) so the estimator never silently returns
    zero and disables the prominence threshold entirely."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 1e-12:
        return _MAD_TO_STD * mad
    return float(np.std(x))


def estimate_scale(
    z: np.ndarray,
    fs_hz: float,
    *,
    method: str = DEFAULT_SCALE_METHOD,
    window_s: float = DEFAULT_SCALE_WINDOW_S,
    inflation_ratio: float = DEFAULT_ADAPTIVE_INFLATION_RATIO,
) -> float:
    """Estimate a signal's amplitude scale, used both to set the Stage-1
    prominence threshold and to normalize Stage-2 (RF) features.

    method="std"
        The original global standard deviation. A single artifact/noise
        burst -- or, as found for I03, slow multi-minute DC drift -- inflates
        it for the whole record. Kept only for backward-compatible / A-B
        comparison runs.
    method="mad"
        Global robust scale (1.4826 * MAD). Resistant to a *few* extreme
        samples, but still one record-wide number, so a long noisy segment
        can still dominate it.
    method="windowed_mad"
        Splits the record into `window_s`-second blocks, computes a robust
        (MAD) scale per block, and takes the *median across blocks*. A
        minority of corrupted blocks can no longer set the threshold for
        the whole record.

        Caveat found during real-data validation: MAD is only robust to a
        *minority* of "outlier" samples per window. On faster/wider-complex
        records, QRS+T can occupy >50% of an 8s window, which flips what
        the median tracks from the quiet baseline to the QRS/T amplitude
        itself, over-suppressing real beats. See method="windowed_std".
    method="windowed_std" (default, adopted 2026-09-12 for Stage 1 and
        2026-09-13 for Stage 2, after real-data validation -- see the
        module-level comment above for the full comparison)
        Splits the record into `window_s`-second blocks, computes plain std
        per block, and takes the *median across blocks*. Keeps std's
        continuous behavior (no discrete flip once a window's "elevated"
        fraction crosses ~50%, unlike windowed_mad) while still discounting
        a minority of artifact-corrupted windows -- and, since each block is
        short relative to slow multi-minute drift, is not dominated by that
        drift either (this is what fixes I03's Stage-2 suppression).
    method="adaptive"
        Uses "std" by default, falling back to "windowed_std" per-record
        when std(z) / windowed_std(z) > inflation_ratio. Tried and
        REJECTED after real-data validation: performed worse than either
        pure method because the stage-2 RF is trained on whichever
        scale_method generated its training candidates, and adaptive's
        near-never-triggering behavior on MIT-BIH vs. frequently-triggering
        behavior on hard INCART records creates a train/test mismatch.
        Kept only as a cautionary example / for future reference.
    """
    z = np.asarray(z, dtype=float)
    method = str(method).lower()
    if method == "std":
        return float(np.std(z))
    if method == "mad":
        return _mad_scale(z)
    if method in ("windowed_mad", "windowed_std"):
        window = max(1, int(round(fs_hz * window_s)))
        n = z.size
        if n <= window:
            return _mad_scale(z) if method == "windowed_mad" else float(np.std(z))
        min_block = max(4, window // 4)
        local_scales = []
        for start in range(0, n, window):
            seg = z[start:start + window]
            if seg.size < min_block:
                continue
            s = _mad_scale(seg) if method == "windowed_mad" else float(np.std(seg))
            if np.isfinite(s) and s > 0:
                local_scales.append(s)
        if not local_scales:
            return _mad_scale(z) if method == "windowed_mad" else float(np.std(z))
        return float(np.median(local_scales))
    if method == "adaptive":
        global_std = float(np.std(z))
        windowed_std = estimate_scale(z, fs_hz, method="windowed_std", window_s=window_s)
        if windowed_std <= 0 or not np.isfinite(windowed_std):
            return global_std
        if global_std / windowed_std > inflation_ratio:
            return windowed_std
        return global_std
    raise ValueError(f"unknown scale method: {method!r}")


# Backward-compatible alias: this function was originally named
# estimate_stage1_scale (before Stage 2 needed it too). Existing code,
# scripts, and tests reference that name via validation_detectors.py.
estimate_stage1_scale = estimate_scale
