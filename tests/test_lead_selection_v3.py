import math

import pytest

from electrotrace.lead_selection import (
    LEAD_SELECTOR_V3_VERSION,
    STARVATION_ALTERNATE_RATE_BPM,
    STARVATION_PRIMARY_RATE_BPM,
    STARVATION_RATE_RATIO,
    choose_two_lead_channel_v3,
)


def test_v3_constants_are_frozen():
    assert LEAD_SELECTOR_V3_VERSION == "edb-ltafdb-svdb-informed-starvation-rescue-v3"
    assert STARVATION_PRIMARY_RATE_BPM == pytest.approx(30.0)
    assert STARVATION_ALTERNATE_RATE_BPM == pytest.approx(30.0)
    assert STARVATION_RATE_RATIO == pytest.approx(2.0)


def test_v3_switches_only_for_gross_primary_starvation():
    assert choose_two_lead_channel_v3(20.0, 80.0) == 1
    assert choose_two_lead_channel_v3(29.9, 60.1) == 1
    assert choose_two_lead_channel_v3(30.0, 100.0) == 0
    assert choose_two_lead_channel_v3(20.0, 30.0) == 0
    assert choose_two_lead_channel_v3(20.0, 40.0) == 0
    assert choose_two_lead_channel_v3(20.0, 40.0001) == 1


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, -math.inf])
def test_v3_rejects_invalid_rates(bad):
    with pytest.raises(ValueError):
        choose_two_lead_channel_v3(bad, 80.0)
    with pytest.raises(ValueError):
        choose_two_lead_channel_v3(20.0, bad)


def test_v3_rejects_non_discriminating_ratio():
    with pytest.raises(ValueError, match="must be > 1"):
        choose_two_lead_channel_v3(20.0, 80.0, minimum_rate_ratio=1.0)
