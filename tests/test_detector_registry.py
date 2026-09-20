from electrotrace.detectors import get_detector


def test_builtin_detector_registry_contains_classical_detectors():
    assert get_detector("pan-tompkins").name == "pan-tompkins"
    assert get_detector("hamilton").name == "hamilton"
    assert get_detector("stage1").name == "stage1"
    assert get_detector("gqrs").name == "wfdb-gqrs"


def test_detector_alias_is_normalized():
    assert get_detector("pan_tompkins").name == "pan-tompkins"
