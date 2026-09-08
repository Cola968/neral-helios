from context_dev50 import KEYS, build_cases, validate_suite


def test_context_dev_is_deterministic_and_disjoint_from_v5_keys():
    a = build_cases(); b = build_cases()
    assert a == b
    assert len(a) == 12
    frozen = {"ORBIT-7K4Q", "NOVA-3M8P", "LUMEN-5R2X", "VECTOR-9C6T"}
    assert not (set(KEYS) & frozen)
    assert all("Kennwort:" not in row["prefix"] for row in a)
    assert all("Passcode:" not in row["prefix"] for row in a)


def test_context_dev_has_single_multi_and_both_languages():
    report = validate_suite()
    assert report["single"] == 8
    assert report["multi"] == 4
    assert report["languages"] == {"de": 6, "en": 6}
    assert report["frozen_v5_key_overlap"] == 0
