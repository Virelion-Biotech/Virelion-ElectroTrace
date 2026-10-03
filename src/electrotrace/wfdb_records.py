"""Shared WFDB record + reference-annotation loading with an explicit audit trail.

Why this exists
---------------
Several validation scripts each carried their own copy of "load the record, read
the ``atr`` annotations, raise on any negative sample index".  Seven of the 75
INCART records trip that check, and every script silently dropped them, so no
reported INCART result covers more than 68 records.  This module centralises the
loading and turns "skip the record" into an auditable decision.

Design rules
------------
* Default behaviour is unchanged: ``policy="error"`` raises exactly where the
  old scripts raised (a ``ValueError`` subclass), so historical results stay
  reproducible.
* ``policy="drop_edges"`` may drop *invalid edge annotations only* (negative
  indices at the start, out-of-range indices at the end).  Invalid annotations
  in the interior, or a non-monotonic remainder, always exclude the record.
* A repair is accepted only when an *independent* detector agrees with the
  remaining reference (alignment check).  A misparsed skip that shifts every
  later annotation therefore cannot slip through as "cleaned".
* Every outcome, including exclusions, produces an :class:`AnnotationAudit`
  that scripts write into their JSON report.

``wfdb`` is imported lazily, so importing this module needs only numpy.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .validation import DEFAULT_BEAT_SYMBOLS

POLICIES = ("error", "drop_edges")
DEFAULT_MIN_ALIGNMENT_COVERAGE = 0.5


@dataclass(frozen=True)
class AnnotationAudit:
    """What was found in one record's reference annotations and what was done."""

    record: str
    action: str  # "kept_unchanged" | "kept_after_dropping_edges" | "excluded"
    reason: str | None
    policy: str
    signal_length: int
    fs_hz: float
    n_annotations_total: int
    n_beat_annotations: int
    n_negative_beat: int
    n_beyond_end_beat: int
    n_leading_invalid: int
    n_trailing_invalid: int
    n_interior_invalid: int
    raw_beat_order_strictly_increasing: bool
    first_invalid: list = field(default_factory=list)  # [[sample, symbol], ...]
    alignment_checked: bool = False
    alignment_coverage: float | None = None
    alignment_median_offset_ms: float | None = None
    n_reference_kept: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class RecordExcluded(ValueError):
    """Raised when a record's reference cannot be used; carries the audit."""

    def __init__(self, record: str, reason: str, audit: AnnotationAudit):
        super().__init__(f"{record}: {reason}")
        self.record = record
        self.reason = reason
        self.audit = audit


@dataclass(frozen=True)
class AnnotatedRecord:
    record: str
    fs_hz: float
    signal: np.ndarray | None
    reference: np.ndarray
    audit: AnnotationAudit


def estimate_alignment(
    peaks: Sequence[int],
    reference: Sequence[int],
    fs_hz: float,
    *,
    tolerance_ms: float = 75.0,
) -> tuple[float, float | None]:
    """Compare an independent detector's peaks with a reference.

    Returns ``(coverage, median_offset_ms)``: the fraction of reference beats
    with any detected peak within ``tolerance_ms``, and the median signed offset
    (detected minus reference, ms) of those nearest matches.  Reference shifted
    by a non-multiple of the RR interval gives low coverage (about
    ``2 * tolerance / RR`` for random phase, roughly 0.2 at 75 ms and 800 ms RR).
    """
    p = np.sort(np.asarray(peaks, dtype=np.int64))
    r = np.asarray(reference, dtype=np.int64)
    if p.size == 0 or r.size == 0:
        return 0.0, None
    idx = np.searchsorted(p, r)
    left = p[np.clip(idx - 1, 0, p.size - 1)]
    right = p[np.clip(idx, 0, p.size - 1)]
    d_left = left - r
    d_right = right - r
    nearest = np.where(np.abs(d_left) <= np.abs(d_right), d_left, d_right)
    tol = float(tolerance_ms) * float(fs_hz) / 1000.0
    within = np.abs(nearest) <= tol
    coverage = float(np.mean(within))
    offset = float(np.median(nearest[within]) * 1000.0 / fs_hz) if within.any() else None
    return coverage, offset


def _resolve_peaks(peaks: Sequence[int] | Callable[[], Sequence[int]] | None) -> np.ndarray | None:
    if peaks is None:
        return None
    value = peaks() if callable(peaks) else peaks
    return np.asarray(value, dtype=np.int64)


def clean_reference_annotations(
    samples: Sequence[int],
    symbols: Sequence[str],
    *,
    n_samples: int,
    fs_hz: float,
    record: str = "",
    beat_symbols: Iterable[str] | None = None,
    policy: str = "error",
    alignment_peaks: Sequence[int] | Callable[[], Sequence[int]] | None = None,
    tolerance_ms: float = 75.0,
    min_alignment_coverage: float = DEFAULT_MIN_ALIGNMENT_COVERAGE,
) -> tuple[np.ndarray, AnnotationAudit]:
    """Validate (and, if the policy allows, repair) beat annotations.

    ``alignment_peaks`` may be an array or a zero-argument callable; the callable
    is only invoked when a repair actually needs verifying, so clean records pay
    nothing.  Raises :class:`RecordExcluded` (a ``ValueError``) when the record
    cannot be used under ``policy``.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    allowed = set(DEFAULT_BEAT_SYMBOLS if beat_symbols is None else beat_symbols)
    samples_arr = np.asarray(samples, dtype=np.int64)
    symbols_arr = np.asarray(list(symbols), dtype=object)
    if samples_arr.shape != symbols_arr.shape:
        raise ValueError("samples and symbols must have the same length")

    is_beat = np.array([s in allowed for s in symbols_arr], dtype=bool)
    beat = samples_arr[is_beat]
    beat_symbols_kept = symbols_arr[is_beat]

    negative = beat < 0
    beyond = beat >= int(n_samples)
    invalid = negative | beyond
    n_invalid = int(invalid.sum())

    # Edge structure: leading run of negatives, trailing run of beyond-end values.
    leading = 0
    while leading < beat.size and negative[leading]:
        leading += 1
    trailing = 0
    while trailing < beat.size - leading and beyond[beat.size - 1 - trailing]:
        trailing += 1
    interior = n_invalid - leading - trailing

    valid = beat[~invalid]
    strictly_increasing = bool(np.all(np.diff(valid) > 0)) if valid.size > 1 else True
    raw_increasing = bool(np.all(np.diff(beat) > 0)) if beat.size > 1 else True

    first_invalid = [
        [int(beat[i]), str(beat_symbols_kept[i])] for i in np.flatnonzero(invalid)[:5]
    ]

    def audit(action, reason, *, checked=False, coverage=None, offset=None, kept=0):
        return AnnotationAudit(
            record=record,
            action=action,
            reason=reason,
            policy=policy,
            signal_length=int(n_samples),
            fs_hz=float(fs_hz),
            n_annotations_total=int(samples_arr.size),
            n_beat_annotations=int(beat.size),
            n_negative_beat=int(negative.sum()),
            n_beyond_end_beat=int(beyond.sum()),
            n_leading_invalid=int(leading),
            n_trailing_invalid=int(trailing),
            n_interior_invalid=int(interior),
            raw_beat_order_strictly_increasing=raw_increasing,
            first_invalid=first_invalid,
            alignment_checked=checked,
            alignment_coverage=coverage,
            alignment_median_offset_ms=offset,
            n_reference_kept=int(kept),
        )

    def exclude(reason, **kw):
        a = audit("excluded", reason, **kw)
        raise RecordExcluded(record, reason, a)

    if policy == "error":
        # Historical behaviour, preserved so locked results reproduce: only a
        # negative index (or non-increasing order) is an error; annotations
        # past the end of the signal were never rejected.
        if negative.any():
            exclude("negative annotation sample index")
        if not raw_increasing:
            exclude("beat annotations not strictly increasing")
        return beat.astype(np.int64), audit("kept_unchanged", None, kept=beat.size)

    if n_invalid == 0 and strictly_increasing:
        return valid.astype(np.int64), audit("kept_unchanged", None, kept=valid.size)
    if n_invalid == 0:
        exclude("beat annotations not strictly increasing")

    # policy == "drop_edges"
    if interior > 0:
        exclude("invalid annotations in the interior of the record")
    if not strictly_increasing:
        exclude("beat annotations not strictly increasing after dropping edges")
    if valid.size == 0:
        exclude("no valid beat annotations remain")

    peaks = _resolve_peaks(alignment_peaks)
    if peaks is None or peaks.size == 0:
        exclude("repair not verified: no independent detector output supplied")
    coverage, offset = estimate_alignment(
        peaks, valid, fs_hz, tolerance_ms=tolerance_ms
    )
    if coverage < float(min_alignment_coverage):
        exclude(
            "repair rejected: reference does not align with independent detector "
            f"(coverage {coverage:.2f} < {min_alignment_coverage:.2f})",
            checked=True,
            coverage=coverage,
            offset=offset,
        )
    return valid.astype(np.int64), audit(
        "kept_after_dropping_edges",
        None,
        checked=True,
        coverage=coverage,
        offset=offset,
        kept=valid.size,
    )


def summarize_audits(audits: Iterable[AnnotationAudit]) -> dict:
    """Counts suitable for the top level of a validation report."""
    audits = list(audits)
    out = {
        "records": len(audits),
        "kept_unchanged": 0,
        "kept_after_dropping_edges": 0,
        "excluded": 0,
        "excluded_records": [],
        "repaired_records": [],
    }
    for a in audits:
        out[a.action] += 1
        if a.action == "excluded":
            out["excluded_records"].append({"record": a.record, "reason": a.reason})
        elif a.action == "kept_after_dropping_edges":
            out["repaired_records"].append(
                {
                    "record": a.record,
                    "dropped": a.n_leading_invalid + a.n_trailing_invalid,
                    "alignment_coverage": a.alignment_coverage,
                }
            )
    return out


def _default_alignment_peaks(signal: np.ndarray, fs_hz: float) -> Callable[[], np.ndarray]:
    def run() -> np.ndarray:
        from .baseline_detectors import pan_tompkins_r_peaks

        return pan_tompkins_r_peaks(signal, fs_hz)

    return run


def load_annotated_record(
    base: str | Path,
    *,
    channel: int = 0,
    extension: str = "atr",
    beat_symbols: Iterable[str] | None = None,
    tolerance_ms: float = 75.0,
    policy: str = "error",
    load_signal: bool = True,
    alignment_peaks: Sequence[int] | Callable[[], Sequence[int]] | None = None,
    min_alignment_coverage: float = DEFAULT_MIN_ALIGNMENT_COVERAGE,
) -> AnnotatedRecord:
    """Load one WFDB record plus validated reference beats.

    Signal handling matches the historical scripts (``physical=False`` and
    ``p_signal`` when present, else ``d_signal``) so numbers do not move.  With
    ``load_signal=False`` only the header is read; then a repair can only be
    verified through ``alignment_peaks`` supplied by the caller.
    """
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - exercised only without wfdb
        raise RuntimeError("wfdb is required to load WFDB records") from exc

    base = str(base)
    name = Path(base).name
    signal = None
    if load_signal:
        rec = wfdb.rdrecord(base, channels=[int(channel)], physical=False)
        signal = np.asarray(
            rec.p_signal[:, 0] if rec.p_signal is not None else rec.d_signal[:, 0],
            dtype=float,
        )
        fs_hz = float(rec.fs)
        n_samples = int(signal.size)
    else:
        header = wfdb.rdheader(base)
        fs_hz = float(header.fs)
        n_samples = int(header.sig_len)

    ann = wfdb.rdann(base, extension)
    if alignment_peaks is None and signal is not None:
        alignment_peaks = _default_alignment_peaks(signal, fs_hz)

    reference, audit = clean_reference_annotations(
        ann.sample,
        ann.symbol,
        n_samples=n_samples,
        fs_hz=fs_hz,
        record=name,
        beat_symbols=beat_symbols,
        policy=policy,
        alignment_peaks=alignment_peaks,
        tolerance_ms=tolerance_ms,
        min_alignment_coverage=min_alignment_coverage,
    )
    return AnnotatedRecord(
        record=name, fs_hz=fs_hz, signal=signal, reference=reference, audit=audit
    )


def raw_annotation_dump(
    base: str | Path, *, extension: str = "atr", head: int = 12
) -> dict:
    """Unfiltered view of a record's annotation file, for root-cause diagnosis."""
    import wfdb

    ann = wfdb.rdann(str(base), extension)
    sample = np.asarray(ann.sample, dtype=np.int64)
    symbol = list(ann.symbol)
    neg = np.flatnonzero(sample < 0)
    diffs = np.diff(sample) if sample.size > 1 else np.array([], dtype=np.int64)
    return {
        "record": Path(str(base)).name,
        "n_annotations": int(sample.size),
        "min_sample": int(sample.min()) if sample.size else None,
        "max_sample": int(sample.max()) if sample.size else None,
        "first": [[int(s), str(y)] for s, y in zip(sample[:head], symbol[:head])],
        "negative_positions": [int(i) for i in neg[:20]],
        "negative_symbols": sorted({str(symbol[i]) for i in neg}),
        "n_negative": int(neg.size),
        "n_nonincreasing_steps": int(np.sum(diffs <= 0)),
        "symbol_counts": {
            str(s): int(symbol.count(s)) for s in sorted(set(symbol))
        },
    }
