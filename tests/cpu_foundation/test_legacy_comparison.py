"""Regression = failing on candidate but not baseline in identical CPU CI."""
from scripts.neral_1_2_legacy_compare import compare_failures

def test_identical_legacy_failures_are_baseline_not_regressions():
    a={"tests":1900,"failures":["old::failure","old::missing_model"],"errors":2}
    b={"tests":1950,"failures":["old::failure","old::missing_model"],"errors":2}
    report=compare_failures(a,b)
    assert report["baseline_failures"]==2
    assert report["new_regressions"]==[]
    assert report["accepted"] is True

def test_new_failure_detected_even_when_old_failures_disappear():
    a={"tests":1900,"failures":["old::failure"],"errors":1}
    b={"tests":1950,"failures":["new::failure"],"errors":1}
    report=compare_failures(a,b)
    assert report["new_regressions"]==["new::failure"]
    assert report["fixed_baseline_failures"]==["old::failure"]
    assert report["accepted"] is False
