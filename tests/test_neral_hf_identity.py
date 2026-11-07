"""Helios 7B model-integrity regression on tiny synthetic snapshots (no GPU)."""
from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from neral_hf_identity import (
    BASE_MODEL, BASE_REVISION, HF_CONFIG, CHAT_TEMPLATE,
    validate_hf_snapshot, validate_adapter,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(root):
    base=root/"base"
    base.mkdir()
    payload={
        "config.json":json.dumps(HF_CONFIG),
        "tokenizer.json":'{"version":"1.0"}',
        "tokenizer_config.json":'{"tokenizer_class":"Qwen2TokenizerFast"}',
        "model.safetensors":"synthetic-no-model-data",
    }
    for name, content in payload.items():
        (base/name).write_text(content,encoding="utf8")
    manifest={"base_model":BASE_MODEL,"revision":BASE_REVISION,
              "files":{name:sha(base/name) for name in payload}}
    (base/"helios-manifest.json").write_text(json.dumps(manifest),encoding="utf8")
    snap=validate_hf_snapshot(base,verify_hashes=True)
    adapter=root/"adapter"
    adapter.mkdir()
    cfg={
        "peft_type":"LORA","task_type":"CAUSAL_LM",
        "r":16,"lora_alpha":16,"lora_dropout":0,
        "bias":"none","init_lora_weights":True,
        "target_modules":["q_proj","k_proj","v_proj","o_proj",
                          "gate_proj","up_proj","down_proj"],
    }
    (adapter/"adapter_model.safetensors").write_bytes(b"synthetic-adapter")
    (adapter/"adapter_config.json").write_text(json.dumps(cfg),encoding="utf8")
    seal={
        "schema":1,"product":"NERAL","version":"5.0.0-pilot.1",
        "base_model":BASE_MODEL,"base_revision":BASE_REVISION,
        "base_snapshot_integrity":snap["sha256"],
        "dataset_sha256":"a"*64,"release_validated":False,
        "files":{name:sha(adapter/name) for name in (
            "adapter_config.json","adapter_model.safetensors")},
    }
    (adapter/"neral-adapter.json").write_text(json.dumps(seal),encoding="utf8")
    return base,adapter


class LocalHFIdentityTests(unittest.TestCase):
    def test_reproduces_exact_pinned_model_and_chat_template(self):
        self.assertEqual(BASE_MODEL,"Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(BASE_REVISION,"c03e6d358207e414f1eca0bb1891e29f1db0e242")
        self.assertEqual(HF_CONFIG["hidden_size"],3584)
        self.assertEqual(HF_CONFIG["num_hidden_layers"],28)
        self.assertIn("<|im_start|>",CHAT_TEMPLATE)
        self.assertIn("add_generation_prompt",CHAT_TEMPLATE)

    def test_valid_base_and_adapter_are_accepted_no_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,adapter=seal(Path(tmp))
            snapshot=validate_hf_snapshot(base,verify_hashes=True)
            self.assertEqual(snapshot["revision"],BASE_REVISION)
            self.assertEqual(snapshot["base_model"],BASE_MODEL)
            proof=validate_adapter(adapter,snapshot)
            self.assertFalse(proof["release_validated"])
            self.assertEqual(proof["files"]["adapter_model.safetensors"],
                             sha(adapter/"adapter_model.safetensors"))

    def test_rejects_tampered_base_weights_even_with_correct_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,adapter=seal(Path(tmp))
            (base/"model.safetensors").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                validate_hf_snapshot(base,verify_hashes=True)

    def test_rejects_wrong_foundation_revision_and_adapter_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,adapter=seal(Path(tmp))
            snapshot=validate_hf_snapshot(base,verify_hashes=True)
            bad=dict(snapshot,revision="0"*40)
            with self.assertRaises(ValueError):
                validate_adapter(adapter,bad)
            manifest=json.loads((base/"helios-manifest.json").read_text())
            manifest["revision"]="0"*40
            (base/"helios-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_hf_snapshot(base,verify_hashes=True)

    def test_rejects_mutated_adapter_payload_and_unsafe_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,adapter=seal(Path(tmp))
            snapshot=validate_hf_snapshot(base,verify_hashes=True)
            (adapter/"adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_adapter(adapter,snapshot)
            (adapter/"adapter_model.safetensors").write_bytes(b"synthetic-adapter")
            cfg=json.loads((adapter/"adapter_config.json").read_text())
            cfg["auto_mapping"]={"evil":"module"}
            (adapter/"adapter_config.json").write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                validate_adapter(adapter,snapshot)

    def test_rejects_unsafe_snapshot_extra_files_and_adapter_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base,adapter=seal(root)
            (base/"execute.py").write_text("print('bad')")
            with self.assertRaises(ValueError):
                validate_hf_snapshot(base,verify_hashes=False)
            (base/"execute.py").unlink()
            (adapter/"extra.txt").write_text("bad")
            with self.assertRaises(ValueError):
                validate_adapter(adapter,validate_hf_snapshot(base,verify_hashes=True))

    def test_quarantined_v6_weight_sha_rejected_even_when_seal_is_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, adapter = seal(Path(tmp))
            snapshot = validate_hf_snapshot(base, verify_hashes=True)
            manifest_path = adapter / "neral-adapter.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["adapter_model.safetensors"] = "8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd"
            manifest_path.write_text(json.dumps(manifest))
            genuine_sha = sha
            with patch("neral_hf_identity._sha", side_effect=lambda path:
                       "8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd" if path.name == "adapter_model.safetensors"
                       else genuine_sha(path)):
                with self.assertRaisesRegex(ValueError, "gesperrt"):
                    validate_adapter(adapter, snapshot)

    def test_agent_backend_refuses_quarantined_weight_sha_before_gpu_init(self):
        from neral_agent_hf import HFAdapterLocalBackend
        from neral_agent_autonomy import AgentModelError
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"standin-not-real-weights")
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 16, "peft_type": "LORA"}), encoding="utf-8")
            with patch("neral_agent_hf.hashlib.file_digest",
                       return_value=Mock(hexdigest=lambda: "8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd")):
                with self.assertRaisesRegex(AgentModelError, "gesperrt"):
                    HFAdapterLocalBackend(base, adapter)

    def test_runtime_backend_does_not_import_missing_legacy_module(self):
        code=Path("neral_agent_hf.py").read_text(encoding="utf8")
        self.assertIn("from neral_hf_identity import",code)
        self.assertIn("validate_adapter",code)
        self.assertIn("validate_hf_snapshot",code)
        self.assertNotIn("from model_catalog import validate_hf_snapshot",code)
        self.assertNotIn("from qwen_local_runtime import CHAT_TEMPLATE",code)
        self.assertIn("local_files_only=True",code)
        self.assertIn("HF_HUB_OFFLINE",code)


if __name__=="__main__":
    unittest.main()
