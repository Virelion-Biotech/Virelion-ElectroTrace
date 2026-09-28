"""Experimental label-free multi-lead selection.

The v1 selector was developed after the first prospective European ST-T
Database run exposed a fixed-channel failure mode. EDB is therefore development
data for this selector. The rule MUST NOT be described as prospectively
validated until it succeeds on a different untouched database.

The selector is intentionally conservative: preserve the historical primary
channel unless its retained Stage-2 probability median is below a frozen floor
and an alternate channel has a higher median. No reference annotations or
clinical labels are used at runtime.
"""
from __future__ import annotations

import math

EDB_DEVELOPMENT_PRIMARY_P50_FLOOR = 0.995
LEAD_SELECTOR_VERSION = "edb-informed-retained-probability-v1"


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def choose_two_lead_channel(
    primary_retained_probability_p50: float,
    alternate_retained_probability_p50: float,
    *,
    primary_floor: float = EDB_DEVELOPMENT_PRIMARY_P50_FLOOR,
) -> int:
    """Return 0 for the historical primary channel or 1 for the alternate.

    The alternate is selected only when the primary channel median retained
    Stage-2 probability is below primary_floor and the alternate median is
    strictly higher. Ties and confident primary channels preserve channel 0.

    This function is label-free: callers must compute the two probability
    summaries without consulting reference annotations.
    """
    primary = _validate_probability(
        "primary_retained_probability_p50",
        primary_retained_probability_p50,
    )
    alternate = _validate_probability(
        "alternate_retained_probability_p50",
        alternate_retained_probability_p50,
    )
    floor = _validate_probability("primary_floor", primary_floor)

    return int(primary < floor and alternate > primary)
