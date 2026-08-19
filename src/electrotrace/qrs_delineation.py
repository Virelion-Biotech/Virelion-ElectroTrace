"""Signal-only QRS onset/offset delineation (experimental, frozen v1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

DELINEATOR_VERSION = "qrs-edge-energy-v1"
DEFAULT_LOW_HZ = 5.0
DEFAULT_HIGH_HZ = 25.0
DEFAULT_SEARCH_S = 0.16
DEFAULT_SMOOTH_S = 0.012
DEFAULT_ENVELOPE_THRESHOLD = 0.15
DEFAULT_AMPLITUDE_THRESHOLD = 0.10
DEFAULT_SUSTAINED_SAMPLES = 3


@dataclass(frozen=True)
class QRSBoundary:
    onset: int
    offset: int
    center: int
    onset_found: bool
    offset_found: bool

    def to_dict(self) -> dict:
        return {
            "onset": self.onset,
            "offset": self.offset,
            "center": self.center,
            "onset_found": self.onset_found,
            "offset_found": self.offset_found,
        }


def _validate(signal: np.ndarray, fs_hz: float, center: int) -> tuple[np.ndarray, float, int]:
    x = np.asarray(signal, dtype=float)
    fs = float(fs_hz)
    c = int(center)
    if x.ndim != 1 or x.size < 32:
        raise ValueError("signal must be one-dimensional with at least 32 samples")
    if not np.isfinite(x).all():
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs_hz must be positive and finite")
    if c < 0 or c >= x.size:
        raise ValueError("center must be a valid sample index")
    return x, fs, c


def _bandpass(x: np.ndarray, fs_hz: float) -> np.ndarray:
    nyquist = fs_hz / 2.0
    low = min(DEFAULT_LOW_HZ, nyquist * 0.4)
    high = min(DEFAULT_HIGH_HZ, nyquist * 0.9)
    if high <= low or low <= 0:
        return x - float(np.median(x))
    sos = sps.butter(3, [low, high], btype="bandpass", fs=fs_hz, output="sos")
    return sps.sosfiltfilt(sos, x - float(np.median(x)))


def delineate_qrs(
    signal: np.ndarray,
    fs_hz: float,
    center: int,
    *,
    search_s: float = DEFAULT_SEARCH_S,
    envelope_threshold: float = DEFAULT_ENVELOPE_THRESHOLD,
    amplitude_threshold: float = DEFAULT_AMPLITUDE_THRESHOLD,
    sustained_samples: int = DEFAULT_SUSTAINED_SAMPLES,
) -> QRSBoundary:
    """Find QRS onset/offset around a supplied R-peak-like center.

    The method is signal-only and deliberately does not use annotations. It
    combines a 5-25 Hz band-limited waveform with a short-time absolute
    derivative envelope. Boundaries are the nearest sustained crossings of
    both normalized edge energy and normalized band-limited amplitude.
    """
    x, fs, c = _validate(signal, fs_hz, center)
    search = float(search_s)
    et = float(envelope_threshold)
    at = float(amplitude_threshold)
    sustain = int(sustained_samples)
    if not np.isfinite(search) or search <= 0:
        raise ValueError("search_s must be positive and finite")
    if not 0 < et < 1 or not np.isfinite(et):
        raise ValueError("envelope_threshold must be in (0, 1)")
    if not 0 < at < 1 or not np.isfinite(at):
        raise ValueError("amplitude_threshold must be in (0, 1)")
    if sustain < 1:
        raise ValueError("sustained_samples must be >= 1")

    half = max(4, int(round(search * fs)))
    lo = max(0, c - half)
    hi = min(x.size, c + half + 1)
    band = _bandpass(x, fs)
    smooth = max(3, int(round(DEFAULT_SMOOTH_S * fs)))
    derivative = np.abs(np.gradient(band) * fs)
    envelope = np.convolve(derivative, np.ones(smooth) / smooth, mode="same")
    local_env = envelope[lo:hi]
    local_band = band[lo:hi]
    env_peak = max(float(np.max(local_env)), np.finfo(float).eps)
    amp_peak = max(float(np.max(np.abs(local_band))), np.finfo(float).eps)
    env_mask = local_env >= et * env_peak
    amp_mask = np.abs(local_band) >= at * amp_peak
    active = env_mask & amp_mask

    c_local = c - lo

    onset = lo
    onset_found = False
    for k in range(c_local, -1, -1):
        stop = k + 1
        start = max(0, stop - sustain)
        if stop - start == sustain and not np.any(active[start:stop]):
            onset = lo + k
            onset_found = True
            break

    offset = hi - 1
    offset_found = False
    for k in range(c_local, len(active)):
        stop = min(len(active), k + sustain)
        if stop - k == sustain and not np.any(active[k:stop]):
            offset = lo + k
            offset_found = True
            break

    onset = max(0, min(onset, c))
    offset = min(x.size - 1, max(offset, c))
    if offset < onset:
        onset, offset = min(onset, c), max(offset, c)
    return QRSBoundary(
        onset=int(onset),
        offset=int(offset),
        center=int(c),
        onset_found=onset_found,
        offset_found=offset_found,
    )
