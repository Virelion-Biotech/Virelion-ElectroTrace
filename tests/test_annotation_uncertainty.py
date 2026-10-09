from electrotrace.annotation_uncertainty import timing_consensus, detector_tolerance_sweep


def test_disagreement_is_preserved_and_single_annotator_is_not_consensus():
    rows = timing_consensus({"beat-1": {"reader-a": 1.0, "reader-b": 1.04}, "beat-2": {"reader-a": 2.0}})
    assert rows[0]["consensus_time_s"] == 1.02
    assert rows[0]["disagreement_end_s"] == 1.04
    assert not rows[1]["consensus_eligible"]


def test_one_detection_does_not_match_multiple_beats_at_loose_tolerance():
    rows = detector_tolerance_sweep([100], [80, 120], fs_hz=1000, tolerances_ms=[10, 30])
    assert rows[0]["true_positive"] == 0
    assert rows[1]["true_positive"] == 1 and rows[1]["false_negative"] == 1
