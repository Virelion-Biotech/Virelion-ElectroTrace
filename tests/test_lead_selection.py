import math

import pytest

from electrotrace.lead_selection import (
    EDB_DEVELOPMENT_PRIMARY_P50_FLOOR,
    LEAD_SELECTOR_VERSION,
    choose_two_lead_channel,
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
