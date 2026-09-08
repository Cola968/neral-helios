import random

from raw_math50_data import CATEGORIES
from raw_math50_transfer_data import (
    TransferTrainingProblemSampler,
    frozen_prompts,
    normalize,
    sample_transfer_problem,
    validate_transfer_curriculum,
)


def test_transfer_curriculum_has_no_frozen_prompt_overlap():
    report = validate_transfer_curriculum(3000)
    assert report["frozen_prompt_overlap"] == 0
    assert report["unique_prompts"] > 2500
    assert set(report["categories"]) == set(CATEGORIES)
    assert all(report["categories"][c] > 0 for c in CATEGORIES)


def test_transfer_sampler_is_deterministic():
    a = TransferTrainingProblemSampler(seed=123)
    b = TransferTrainingProblemSampler(seed=123)
    rows_a = [a.next() for _ in range(250)]
    rows_b = [b.next() for _ in range(250)]
    assert rows_a == rows_b


def test_core_ranges_are_outside_frozen_id_and_ood_bands():
    rng = random.Random(7)
    for _ in range(100):
        r = sample_transfer_problem(rng, "add_sub", "en")
        _, _, a, _, b = r["canonical"].split(":")
        assert 200 <= int(a) <= 999 and 200 <= int(b) <= 999
        r = sample_transfer_problem(rng, "powers", "de")
        _, _, a, e = r["canonical"].split(":")
        assert 101 <= int(a) <= 200 and int(e) in {2, 3}
        r = sample_transfer_problem(rng, "sequence", "en")
        _, _, start, step = r["canonical"].split(":")
        assert 700 <= int(start) <= 1600 and 51 <= int(step) <= 90


def test_individual_samples_never_equal_frozen_prompts():
    banned = frozen_prompts(); rng = random.Random(99)
    for category in CATEGORIES:
        for language in ("de", "en"):
            for _ in range(100):
                row = sample_transfer_problem(rng, category, language)
                assert normalize(row["prompt"]) not in banned
