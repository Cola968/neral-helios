from __future__ import annotations

import pytest

from hf_candidate_admission50 import assess_manifest
from hf_source_policy50 import SOURCES


def reserved_eval_ids():
    return sorted(k for k, v in SOURCES.items() if v.role == "public-eval-holdout")


def good(dataset_id="HuggingFaceTB/finemath"):
    return {
        "dataset_id": dataset_id,
        "revision": "a" * 40,
        "license": "ODC-By-1.0",
        "content_origin": "web-human-filtered",
        "file_hashes": {"data/chunk.parquet": "b" * 64},
        "cleaning_pipeline_sha256": "c" * 64,
        "benchmark_screen": {
            "benchmark_manifest_sha256": "d" * 64,
            "exact_prompt_matches": 0,
            "near_duplicate_matches": 0,
            "answer_leak_matches": 0,
        },
        "sampled_rows": 10000,
        "external_model_generated_rows": 0,
        "reserved_eval_dataset_ids": reserved_eval_ids(),
    }


def test_finemath_can_pass_audit_but_stays_policy_blocked():
    result = assess_manifest(good())
    assert result["audit_pass"] is True
    assert result["eligible_for_manual_policy_review"] is True
    assert result["training_allowed_by_current_policy"] is False
    assert "do not train" in result["next_action"]


def test_any_benchmark_overlap_quarantines_candidate():
    manifest = good()
    manifest["benchmark_screen"]["near_duplicate_matches"] = 1
    result = assess_manifest(manifest)
    assert result["audit_pass"] is False
    assert "benchmark_contamination_detected" in result["failures"]


def test_external_generated_rows_are_rejected():
    manifest = good()
    manifest["external_model_generated_rows"] = 2
    result = assess_manifest(manifest)
    assert not result["audit_pass"]
    assert "external_model_generated_training_rows" in result["failures"]


def test_public_holdout_can_never_be_admitted():
    manifest = good("openai/gsm8k")
    manifest["license"] = "MIT"
    manifest["content_origin"] = "human-crowdsourced"
    result = assess_manifest(manifest)
    assert not result["audit_pass"]
    assert "public_eval_holdout" in result["failures"]


def test_unknown_source_fails_closed():
    manifest = good("unknown/corpus")
    with pytest.raises(ValueError):
        assess_manifest(manifest)


def test_incomplete_reserved_eval_registry_is_rejected():
    manifest = good()
    manifest["reserved_eval_dataset_ids"] = ["openai/gsm8k"]
    result = assess_manifest(manifest)
    assert "reserved_eval_set_incomplete" in result["failures"]


def test_bad_hashes_are_rejected_without_silently_passing():
    manifest = good()
    manifest["file_hashes"] = {"data/chunk.parquet": "not-a-hash"}
    result = assess_manifest(manifest)
    assert not result["audit_pass"]
    assert "invalid_file_hash" in result["failures"]
