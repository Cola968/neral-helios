"""CPU-only regression tests for Phoenix refresh source/precision contracts."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helios350_preview import CONTINUITY_SHA, TOKENIZER_SHA
from phoenix_continuity_cpu_probe import EXPECTED_TOKENS
from phoenix_foundation_refresh_dev import (
    MAX_TARGET,
    lineage,
    validate_refresh_manifest_contract,
)
from source_lock350 import LOCK_PATH, load_lock
from recovery_identity import file_sha
from train_sft_pilot import base_token_accounting


def good_manifest() -> dict:
    return {
        "schema": 1,
        "architecture": "frontier350-v1",
        "corpus_role": "foundation-pretraining-refresh-development",
        "classification": "DEVELOPMENT_ONLY_NOT_PRODUCTION_APPROVED",
        "continuation_profile": "phoenix-quality-refresh-v2-approved-sources",
        "continuation_from_tokens": EXPECTED_TOKENS,
        "continuation_from_checkpoint_sha256": CONTINUITY_SHA,
        "new_train_tokens": MAX_TARGET,
        "validation_tokens": 500_000,
        "production_eligible": False,
        "historical_2b_document_overlap_checked": False,
        "train_validation_exact_document_disjoint": True,
        "source_lock_sha256": file_sha(LOCK_PATH),
        "finemath_candidate_included": False,
        "benchmark_content_included": None,
        "benchmark_screening_status": "NOT_PERFORMED_BY_THIS_BUILDER_REQUIRES_SEPARATE_AUDIT",
        "source_mix": {
            "fineweb_edu_en": 0.55,
            "fineweb_en": 0.15,
            "fineweb2_de": 0.30,
        },
        "source_policy": {
            "fineweb_edu_en": {
                "dataset_id": "HuggingFaceFW/fineweb-edu",
                "role": "pretrain",
                "license": "ODC-By-1.0",
                "content_origin": "web-human-filtered",
            },
            "fineweb_en": {
                "dataset_id": "HuggingFaceFW/fineweb",
                "role": "pretrain",
                "license": "ODC-By-1.0",
                "content_origin": "web-human",
            },
            "fineweb2_de": {
                "dataset_id": "HuggingFaceFW/fineweb-2",
                "role": "pretrain",
                "license": "ODC-By-1.0",
                "content_origin": "web-human",
            },
        },
        "tokenizer": {"vocab_size": 16_384, "tokenizer_sha256": TOKENIZER_SHA},
    }


class RefreshContractTests(unittest.TestCase):
    def test_current_policy_approved_manifest_contract_passes(self):
        report = validate_refresh_manifest_contract(
            good_manifest(), TOKENIZER_SHA, load_lock()
        )
        self.assertEqual(
            report["source_ids"],
            {
                "HuggingFaceFW/fineweb-edu",
                "HuggingFaceFW/fineweb",
                "HuggingFaceFW/fineweb-2",
            },
        )

    def test_legacy_finemath_refresh_is_blocked(self):
        manifest = good_manifest()
        manifest["continuation_profile"] = "phoenix-quality-refresh-v1"
        manifest["finemath_candidate_included"] = True
        with self.assertRaisesRegex(ValueError, "legacy|unapproved"):
            validate_refresh_manifest_contract(manifest, TOKENIZER_SHA, load_lock())

    def test_finemath_flag_is_blocked_even_under_current_profile(self):
        manifest = good_manifest()
        manifest["finemath_candidate_included"] = True
        with self.assertRaisesRegex(ValueError, "FineMath"):
            validate_refresh_manifest_contract(manifest, TOKENIZER_SHA, load_lock())

    def test_source_policy_receipt_cannot_smuggle_unapproved_family(self):
        manifest = good_manifest()
        manifest["source_policy"]["fineweb_en"] = {
            "dataset_id": "HuggingFaceTB/finemath",
            "role": "future-pretrain-candidate",
            "license": "ODC-By-1.0",
            "content_origin": "web-human-filtered",
        }
        with self.assertRaisesRegex(ValueError, "not currently admitted|exact admitted"):
            validate_refresh_manifest_contract(manifest, TOKENIZER_SHA, load_lock())

    def test_benchmark_cleanliness_cannot_be_claimed_by_builder(self):
        manifest = good_manifest()
        manifest["benchmark_content_included"] = False
        with self.assertRaisesRegex(ValueError, "benchmark cleanliness"):
            validate_refresh_manifest_contract(manifest, TOKENIZER_SHA, load_lock())


    def test_clean_finemath_development_manifest_is_separate_bounded_contract(self):
        manifest=good_manifest()
        manifest["continuation_profile"]="phoenix-quality-refresh-v3-finemath-development"
        manifest["finemath_candidate_included"]=True
        manifest["benchmark_content_included"]=False
        manifest["benchmark_screening_status"]="FULL_STREAM_REJECTION_RESERVED_AND_PUBLIC_DEV_V1"
        manifest["source_mix"]={"fineweb_edu_en":0.55,"fineweb_en":0.10,"fineweb2_de":0.15,"finemath":0.20}
        manifest["source_policy"]["finemath"]={
            "dataset_id":"HuggingFaceTB/finemath","role":"future-pretrain-candidate",
            "license":"ODC-By-1.0","content_origin":"web-human-filtered","development_only":True,
        }
        lock=load_lock()
        manifest["resolved_revisions"]={name:lock["sources"][name]["revision"] for name in manifest["source_mix"]}
        manifest["finemath_candidate_admission_sha256"]="a"*64
        manifest["finemath_candidate_admission"]={
            "dataset_id":"HuggingFaceTB/finemath","revision":lock["sources"]["finemath"]["revision"],
            "audit_pass":True,"eligible_for_manual_policy_review":True,
            "training_allowed_by_current_policy":False,
            "evidence":{"sampled_rows":10_000,"external_model_generated_rows":0,
                        "benchmark_overlap":{"exact_prompt_matches":0,"near_duplicate_matches":0,"answer_leak_matches":0}},
        }
        manifest["full_stream_leakage_screen"]={"documents_screened":1234,"exact_rejections":2,"near_rejections":1}
        report=validate_refresh_manifest_contract(manifest,TOKENIZER_SHA,lock)
        self.assertTrue(report["development_only"])
        self.assertIn("HuggingFaceTB/finemath",report["source_ids"])

    def test_finemath_development_manifest_rejects_dirty_admission(self):
        manifest=good_manifest()
        manifest["continuation_profile"]="phoenix-quality-refresh-v3-finemath-development"
        manifest["finemath_candidate_included"]=True
        manifest["benchmark_content_included"]=False
        manifest["benchmark_screening_status"]="FULL_STREAM_REJECTION_RESERVED_AND_PUBLIC_DEV_V1"
        manifest["source_mix"]={"fineweb_edu_en":0.55,"fineweb_en":0.10,"fineweb2_de":0.15,"finemath":0.20}
        manifest["source_policy"]["finemath"]={
            "dataset_id":"HuggingFaceTB/finemath","role":"future-pretrain-candidate",
            "license":"ODC-By-1.0","content_origin":"web-human-filtered","development_only":True,
        }
        lock=load_lock()
        manifest["resolved_revisions"]={name:lock["sources"][name]["revision"] for name in manifest["source_mix"]}
        manifest["finemath_candidate_admission_sha256"]="a"*64
        manifest["finemath_candidate_admission"]={
            "dataset_id":"HuggingFaceTB/finemath","revision":lock["sources"]["finemath"]["revision"],
            "audit_pass":True,"eligible_for_manual_policy_review":True,
            "training_allowed_by_current_policy":False,
            "evidence":{"sampled_rows":10_000,"external_model_generated_rows":0,
                        "benchmark_overlap":{"exact_prompt_matches":0,"near_duplicate_matches":1,"answer_leak_matches":0}},
        }
        manifest["full_stream_leakage_screen"]={"documents_screened":1234,"exact_rejections":0,"near_rejections":0}
        with self.assertRaisesRegex(ValueError,"contamination|audit"):
            validate_refresh_manifest_contract(manifest,TOKENIZER_SHA,lock)

    def test_sft_token_accounting_preserves_historical_anchor(self):
        refresh = {
            "processed_tokens": 1_048_576,
            "lineage": {
                "kind": "phoenix_foundation_refresh_fp32_master_adamw_v2",
                "historical_processed_tokens": EXPECTED_TOKENS,
                "target_new_tokens": MAX_TARGET,
            },
        }
        row = base_token_accounting(refresh)
        self.assertEqual(row["historical_pretraining_tokens"], EXPECTED_TOKENS)
        self.assertEqual(row["new_foundation_lineage_tokens"], 1_048_576)
        self.assertEqual(
            row["cumulative_reference_tokens"], EXPECTED_TOKENS + 1_048_576
        )

    def test_sft_token_accounting_historical_checkpoint_is_not_double_counted(self):
        row = base_token_accounting({"processed_tokens": EXPECTED_TOKENS})
        self.assertEqual(row["historical_pretraining_tokens"], EXPECTED_TOKENS)
        self.assertEqual(row["new_foundation_lineage_tokens"], 0)
        self.assertEqual(row["cumulative_reference_tokens"], EXPECTED_TOKENS)

    def test_lineage_binds_fp32_master_precision(self):
        row = lineage(
            "a" * 64, MAX_TARGET, 256, 4, 1, 1e-5, 35023, False, "default"
        )
        self.assertEqual(row["kind"], "phoenix_foundation_refresh_fp32_master_adamw_v2")
        self.assertEqual(row["master_parameter_dtype"], "float32")
        self.assertEqual(row["adamw_moment_dtype"], "float32")
        self.assertEqual(row["compute_autocast_dtype"], "bfloat16")
        self.assertEqual(row["compile_mode"], "default")


if __name__ == "__main__":
    unittest.main()
