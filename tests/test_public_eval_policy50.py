from __future__ import annotations

import pytest

from hf_source_policy50 import SOURCES
from prepare_gsm8k_holdout50 import DATASET_ID, UPSTREAM_REVISION, UPSTREAM_SHA256, extract_final_answer


def test_gsm8k_is_permanently_reserved_from_training():
    policy = SOURCES[DATASET_ID]
    assert policy.role == "public-eval-holdout"
    assert policy.allow_training is False


def test_gsm8k_final_answer_extractor_keeps_only_final_marker():
    assert extract_final_answer("work 1 + 2 = 3\n#### 3") == "3"
    assert extract_final_answer("some reasoning\n#### 1,250") == "1,250"
    assert extract_final_answer("reasoning\n#### -4.5") == "-4.5"
    with pytest.raises(ValueError):
        extract_final_answer("no final marker here")


def test_gsm8k_upstream_is_pinned():
    assert len(UPSTREAM_REVISION) == 40
    assert len(UPSTREAM_SHA256) == 64
    assert UPSTREAM_SHA256 == "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
