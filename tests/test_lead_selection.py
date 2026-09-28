import math

import pytest

from electrotrace.lead_selection import (
    EDB_DEVELOPMENT_PRIMARY_P50_FLOOR,
    LEAD_SELECTOR_VERSION,
    LEAD_SELECTOR_V2_VERSION,
    choose_two_lead_channel,
    choose_two_lead_channel_v2,
)


def test_frozen_selector_version_and_floor():
    assert LEAD_SELECTOR_VERSION == "edb-informed-retained-probability-v1"
    assert EDB_DEVELOPMENT_PRIMARY_P50_FLOOR == pytest.approx(0.995)


def test_selector_preserves_confident_primary_even_if_alternate_is_higher():
    assert choose_two_lead_channel(0.9990, 0.9992) == 0


def test_selector_switches_only_for_low_confidence_primary_and_higher_alternate():
    assert choose_two_lead_channel(0.95, 0.99) == 1
    assert choose_two_lead_channel(0.95, 0.94) == 0
    assert choose_two_lead_channel(0.95, 0.95) == 0


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_selector_rejects_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        choose_two_lead_channel(bad, 0.9)
    with pytest.raises(ValueError):
        choose_two_lead_channel(0.9, bad)


def test_selector_rejects_invalid_floor():
    with pytest.raises(ValueError):
        choose_two_lead_channel(0.9, 0.95, primary_floor=1.5)


def test_v2_version_is_frozen():
    assert LEAD_SELECTOR_V2_VERSION == "edb-ltafdb-informed-quality-consensus-v2"


def test_v2_switches_when_all_label_free_quality_indicators_agree():
    # LTAFDB record-105-like exposed development pattern.
    assert choose_two_lead_channel_v2(
        0.6577, 0.6977, 0.3142, 0.5004, 0.0050, 0.4353
    ) == 1
    # LTAFDB record-203-like pattern.
    assert choose_two_lead_channel_v2(
        0.9241, 0.9366, 0.4574, 0.4994, 0.3886, 0.4899
    ) == 1


def test_v2_blocks_known_quality_disagreement_patterns():
    # Record-45-like: alternate p50/QRS improve but retention falls.
    assert choose_two_lead_channel_v2(
        0.3822, 0.6419, 0.0433, 0.3500, 0.6678, 0.4961
    ) == 0
    # Record-53-like: alternate p50/retention improve but QRS fraction falls.
    assert choose_two_lead_channel_v2(
        0.4482, 0.5424, 0.1133, 0.0057, 0.1424, 0.2929
    ) == 0


def test_v2_preserves_primary_on_ties_or_confident_primary():
    assert choose_two_lead_channel_v2(
        0.999, 1.0, 0.2, 0.9, 0.2, 0.9
    ) == 0
    assert choose_two_lead_channel_v2(
        0.9, 0.9, 0.2, 0.9, 0.2, 0.9
    ) == 0
    assert choose_two_lead_channel_v2(
        0.9, 0.95, 0.2, 0.2, 0.2, 0.9
    ) == 0
    assert choose_two_lead_channel_v2(
        0.9, 0.95, 0.2, 0.9, 0.2, 0.2
    ) == 0


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_v2_rejects_invalid_label_free_inputs(bad):
    args = [0.9, 0.95, 0.2, 0.3, 0.4, 0.5]
    for index in range(len(args)):
        changed = list(args)
        changed[index] = bad
        with pytest.raises(ValueError):
            choose_two_lead_channel_v2(*changed)
