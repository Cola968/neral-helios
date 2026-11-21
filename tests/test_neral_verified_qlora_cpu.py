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

class SmokeProofSafety(unittest.TestCase):
    def test_actual_gradient_and_updated_weights_are_required(self):
        from scripts.neral_qlora_verified_v39 import validate_smoke_proof
        good={'grad_norm':0.52,'training_loss':1.23,'global_step':1,'peak_allocated_gib':9.61,'peak_reserved_gib':10.11,'nvidia_peak_mib':10500,'adapter_before':'a','adapter_after':'b'}
        self.assertTrue(validate_smoke_proof(good))
        for field,value in (('grad_norm',0),('grad_norm',float('nan')),('training_loss',float('inf')),('global_step',0),('peak_reserved_gib',11),('nvidia_peak_mib',11500),('adapter_after','a')):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_smoke_proof({**good,field:value})

    def test_one_step_smoke_does_not_need_frozen_holdout(self):
        from unittest.mock import patch
        from scripts.neral_qlora_verified_v39 import main
        with tempfile.TemporaryDirectory() as d:
            cp=Path(d)/'checkpoint';cp.mkdir()
            with patch('scripts.neral_qlora_verified_v39.gpu_train') as train:
                self.assertEqual(main(['--run-gpu','--train',str(SEED),'--checkpoint',str(cp),'--out',str(Path(d)/'adapter'),'--max-steps','1','--context','1024']),0)
                self.assertEqual(train.call_count,1)
            with self.assertRaises(SystemExit):
                main(['--run-gpu','--train',str(SEED),'--checkpoint',str(cp),'--out',str(Path(d)/'full')])
