"""Beat-level ECG segmentation from detected R peaks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Beat:
    index: int
    r_index: int
    r_time: float
    start: float
    end: float
    rr_prev_s: float | None
    rr_next_s: float | None
    heart_rate_bpm: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def segment_beats(time: np.ndarray, peaks: np.ndarray, pre_s: float = 0.35, post_s: float = 0.55) -> list[Beat]:
    time = np.asarray(time, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("time must be a one-dimensional array with at least two samples")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    if len(peaks) == 0:
        return []
    if not np.isfinite(pre_s) or not np.isfinite(post_s) or pre_s <= 0 or post_s <= 0:
        raise ValueError("beat windows must be positive and finite")
    peaks = np.unique(peaks[(peaks >= 0) & (peaks < len(time))])
    if len(peaks) == 0:
        return []
    r_times = time[peaks]
    beats: list[Beat] = []
    for i, (idx, rt) in enumerate(zip(peaks, r_times)):
        prev_rr = float(rt - r_times[i - 1]) if i else None
        next_rr = float(r_times[i + 1] - rt) if i + 1 < len(r_times) else None
        rr = prev_rr if prev_rr is not None and prev_rr > 0 else next_rr
        hr = 60.0 / rr if rr is not None and rr > 0 else None
        start = max(float(time[0]), float(rt - pre_s))
        end = min(float(time[-1]), float(rt + post_s))
        beats.append(Beat(i, int(idx), float(rt), start, end, prev_rr, next_rr, hr))
    return beats
