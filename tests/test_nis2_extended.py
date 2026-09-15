from benchmark_nis2_extended50 import PAIR_CASES, SUITE_VERSION, suite_sha256


def test_nis2_extended_is_separate_and_stable_shape():
    assert SUITE_VERSION == "nis2-extended-v1"
    assert len(PAIR_CASES) == 40
    categories = {row[0] for row in PAIR_CASES}
    assert categories == {
        "multi_step_math",
        "code_debugging",
        "context_conflict",
        "uncertainty_calibration",
        "dependency_chain",
    }
    counts = {k: sum(1 for row in PAIR_CASES if row[0] == k) for k in categories}
    assert set(counts.values()) == {8}
    assert len({(row[1], row[2], row[3]) for row in PAIR_CASES}) == len(PAIR_CASES)
    assert len(suite_sha256()) == 64


def test_nis2_core_contract_is_not_modified_by_extended_suite():
    from benchmark_nis2_chat50 import EXACT_GENERATION_CASES, PAIR_CASES as CORE_CASES, SUITE_VERSION as CORE_VERSION

    assert CORE_VERSION == "nis2-chat-core-v1"
    assert len(CORE_CASES) == 40
    assert len(EXACT_GENERATION_CASES) == 8
