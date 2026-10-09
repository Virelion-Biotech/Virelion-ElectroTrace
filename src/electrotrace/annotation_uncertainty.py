"""Event-aligned multi-annotator timing uncertainty and detector tolerance sweeps."""

from __future__ import annotations
import math
import numpy as np
from .validation import match_peaks


def timing_consensus(events, *, minimum_annotators=2):
    """Caller-supplied event IDs align annotators; no uncertain beat pairing is guessed.

    Each event maps independent annotator IDs to timing in seconds. The interval
    is observed disagreement, not a confidence interval for a physiological truth.
    """
    if isinstance(minimum_annotators, bool) or not isinstance(minimum_annotators, int) or minimum_annotators < 2:
        raise ValueError("minimum_annotators must be an integer >=2")
    if not isinstance(events, dict) or not events:
        raise ValueError("Event-aligned annotation mappings are required")
    output = []
    for event, annotators in events.items():
        if not str(event).strip() or not isinstance(annotators, dict) or not annotators:
            raise ValueError("Every event needs a nonempty annotator mapping")
        values = []
        for annotator, value in annotators.items():
            if (
                not str(annotator).strip()
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Annotator IDs and nonnegative finite times in seconds are required")
            values.append(value)
        output.append(
            {
                "event_id": event,
                "consensus_time_s": float(np.median(values)),
                "disagreement_start_s": min(values),
                "disagreement_end_s": max(values),
                "disagreement_width_s": max(values) - min(values),
                "n_annotators": len(values),
                "consensus_eligible": len(values) >= minimum_annotators,
                "annotators": dict(annotators),
                "interval_kind": "observed_annotator_disagreement",
            }
        )
    return output


def detector_tolerance_sweep(detected_samples, reference_samples, *, fs_hz, tolerances_ms):
    tolerances = tuple(tolerances_ms)
    if not tolerances or len(set(tolerances)) != len(tolerances):
        raise ValueError("Specify unique prespecified timing tolerances")
    return [
        {"tolerance_ms": float(t), **match_peaks(detected_samples, reference_samples, fs_hz, t).to_dict()}
        for t in sorted(tolerances)
    ]
