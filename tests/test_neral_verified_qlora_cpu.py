"""No CUDA, model download or GPU benchmark invoked by these tests."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.neral_qlora_verified_v39 import (
    RANK, ALPHA, MAX_GPU_ALLOC_GIB, cpu_plan, load_verified_dataset,
    assert_disjoint_holdout, main)
from scripts.neral_v39_prepare_local_dataset import merge_seed

SEED = Path(__file__).resolve().parents[1]/"datasets"/"neral_v39_seed97.jsonl"

class VerifiedQloraCPU(unittest.TestCase):
    def test_seed_has_exactly_97_unique_grader_checked_rows(self):
        rows=load_verified_dataset(SEED)
        self.assertEqual(len(rows),97)
        self.assertEqual({r["meta"]["source"] for r in rows},{"v39_benchmark"})
        self.assertTrue(all(r["meta"]["verification_kind"]=="benchmark_strict_e2e"
                            for r in rows))
        self.assertTrue(all(r["meta"]["verifier_ok"] is False for r in rows))

    def test_r16_nf4_gradient_checkpointing_and_explicit_vram_caveat(self):
        self.assertEqual(RANK,16)
        self.assertEqual(ALPHA,32)
        plan=cpu_plan(load_verified_dataset(SEED),1024)
        self.assertEqual(plan["quantization"],"4bit_nf4_double_quant")
        self.assertTrue(plan["gradient_checkpointing"])
        self.assertLess(MAX_GPU_ALLOC_GIB,11.0)
        self.assertFalse(plan["vram_below_11gb_guaranteed_by_cpu_test"])
        with self.assertRaises(ValueError):cpu_plan([],2048)

    def test_disjoint_holdout_refuses_reuse_of_benchmark(self):
        rows=load_verified_dataset(SEED)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"holdout.jsonl"
            path.write_text(json.dumps({"question":rows[0]["instruction"]})+"\n")
            with self.assertRaisesRegex(ValueError,"overlaps"):
                assert_disjoint_holdout(rows,path)
            path.write_text(json.dumps({"question":"Brandneuer unbenutzter Holdout"})+"\n")
            assert_disjoint_holdout(rows,path)

    def test_idempotent_merge_keeps_separate_user_data_in_gitignored_file(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/"data"/"verified_training_dataset.jsonl"
            self.assertEqual(merge_seed(SEED,target),97)
            self.assertEqual(merge_seed(SEED,target),97)
            self.assertEqual(len(target.read_text().splitlines()),97)

    def test_cpu_only_entrypoint_no_model_import(self):
        self.assertEqual(main(["--train",str(SEED),"--check-cpu"]),0)

if __name__=="__main__":unittest.main()
