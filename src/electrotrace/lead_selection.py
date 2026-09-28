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
LEAD_SELECTOR_V2_VERSION = "edb-ltafdb-informed-quality-consensus-v2"


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


def choose_two_lead_channel_v2(
    primary_retained_probability_p50: float,
    alternate_retained_probability_p50: float,
    primary_retained_qrs_band_fraction: float,
    alternate_retained_qrs_band_fraction: float,
    primary_retention_fraction: float,
    alternate_retention_fraction: float,
    *,
    primary_floor: float = EDB_DEVELOPMENT_PRIMARY_P50_FLOOR,
) -> int:
    """Return a conservative quality-consensus lead choice.

    Selector v2 was developed only after EDB and LTAFDB were exposed. It
    preserves channel 0 unless all three label-free quality indicators agree
    in favor of channel 1:

    * primary retained-probability p50 is below the frozen 0.995 floor;
    * alternate retained-probability p50 is strictly higher;
    * alternate median retained QRS-band fraction is strictly higher;
    * alternate Stage-2 retention fraction is strictly higher.

    QRS-band and retention fractions are validated in [0, 1]. Ties,
    disagreement, or a confident primary channel preserve channel 0.
    """
    primary_p50 = _validate_probability(
        "primary_retained_probability_p50",
        primary_retained_probability_p50,
    )
    alternate_p50 = _validate_probability(
        "alternate_retained_probability_p50",
        alternate_retained_probability_p50,
    )
    primary_qrs = _validate_probability(
        "primary_retained_qrs_band_fraction",
        primary_retained_qrs_band_fraction,
    )
    alternate_qrs = _validate_probability(
        "alternate_retained_qrs_band_fraction",
        alternate_retained_qrs_band_fraction,
    )
    primary_retention = _validate_probability(
        "primary_retention_fraction",
        primary_retention_fraction,
    )
    alternate_retention = _validate_probability(
        "alternate_retention_fraction",
        alternate_retention_fraction,
    )
    floor = _validate_probability("primary_floor", primary_floor)

    return int(
        primary_p50 < floor
        and alternate_p50 > primary_p50
        and alternate_qrs > primary_qrs
        and alternate_retention > primary_retention
    )
