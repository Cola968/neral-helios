"""Contracts for the rented-GPU QLoRA experiment; no weights or cloud required."""
import unittest
from pathlib import Path
from scripts.neral_vast_cloud import (
    MODEL_ID, BASE_REVISION, REQUIRED_FILES, EXTRAS, select_file,
    validate_experiment, check_device,
)

class VastCloudContractTests(unittest.TestCase):
    def test_pinned_model_and_only_safetensors(self):
        self.assertEqual(MODEL_ID,"Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(BASE_REVISION,"c03e6d358207e414f1eca0bb1891e29f1db0e242")
        for name in REQUIRED_FILES:
            self.assertTrue(select_file(name),name)
        self.assertTrue(select_file("model-00001-of-00004.safetensors"))
        for name in ("pytorch_model.bin","modeling_qwen2.py","evil.sh",
                     "../../secrets.txt",".cache/huggingface/token","adapter_model.safetensors"):
            self.assertFalse(select_file(name),name)
        self.assertIn("model.safetensors.index.json",EXTRAS)

    def test_experiment_profile_is_explicit_and_bounded(self):
        self.assertEqual(validate_experiment("smoke",16,3,512),(16,3,512))
        self.assertEqual(validate_experiment("train",32,480,512),(32,480,512))
        for mode,rank,steps,tokens in (
            ("smoke",32,3,512),("train",16,480,512),
            ("train",32,120,512),("train",32,480,1024),
            ("train",64,480,512),("train",32,481,512),
            ("train",32,480,2048),("prepare",32,480,512),
        ):
            with self.subTest(profile=(mode,rank,steps,tokens)):
                with self.assertRaises(ValueError):
                    validate_experiment(mode,rank,steps,tokens)

    def test_gpu_gate_rejects_wrong_device_and_busy_gpu(self):
        self.assertTrue(check_device("NVIDIA A100-SXM4-40GB",40*1024**3,34*1024**3))
        with self.assertRaises(ValueError):
            check_device("NVIDIA GeForce RTX 4070",12*1024**3,11*1024**3)
        with self.assertRaises(ValueError):
            check_device("NVIDIA A100 PCIe",40*1024**3,26*1024**3)
        with self.assertRaises(ValueError):
            check_device("NVIDIA A100 PCIe",40*1024**3,-1)

    def test_scripts_are_noninteractive_and_no_automatic_billing(self):
        readme=Path("docs/NERAL_VAST_A100.md").read_text(encoding="utf8")
        setup=Path("scripts/neral_vast_setup.sh").read_text(encoding="utf8")
        self.assertIn("bash scripts/neral_vast_setup.sh",readme)
        self.assertIn("destroy",readme.lower())
        self.assertIn("torch==2.8.0+cu128",setup)
        self.assertNotIn("vastai create",setup)
        self.assertNotIn("api_key=",setup.lower())
        self.assertIn("SMOKE",readme)
        self.assertIn("40 GB",readme)

if __name__=="__main__":
    unittest.main()
