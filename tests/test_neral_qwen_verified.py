"""Guard the shipped Helios HF adapter against missing pinned-helper imports."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from neral_qwen_verified import (
    HF_BASE, HF_CONFIG, BASE_REVISION, CHAT_TEMPLATE,
    validate_hf_snapshot,
)


class QwenVerifiedRuntimeTests(unittest.TestCase):
    def _snapshot(self, root: Path):
        root.mkdir()
        files = {
            "config.json": json.dumps(HF_CONFIG).encode(),
            "tokenizer.json": b"{}",
            "tokenizer_config.json": b'{"tokenizer_class":"Qwen2TokenizerFast"}',
            "special_tokens_map.json": b"{}",
            "model.safetensors": b"synthetic test fixture; never deserialize",
        }
        for name, raw in files.items():
            (root / name).write_bytes(raw)
        self._rebuild_manifest(root)
        return root

    def _rebuild_manifest(self, root, *, revision=BASE_REVISION):
        data = {
            "base_model": HF_BASE, "revision": revision,
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.iterdir() if p.name != "helios-manifest.json"
            }
        }
        (root / "helios-manifest.json").write_text(json.dumps(data),encoding="utf8")

    def test_pinned_manifest_sha256_and_template(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._snapshot(Path(d)/"model")
            checked=validate_hf_snapshot(root, verify_hashes=True)
            self.assertEqual(checked["revision"],BASE_REVISION)
            self.assertEqual(checked["base_model"],HF_BASE)
            self.assertEqual(len(checked["sha256"]),64)
        self.assertIn("<|im_start|>", CHAT_TEMPLATE)
        self.assertIn("<|im_end|>", CHAT_TEMPLATE)
        self.assertIn("add_generation_prompt", CHAT_TEMPLATE)
        self.assertNotIn("import ",CHAT_TEMPLATE)

    def test_rejects_wrong_revision_even_if_signed_manifest_is_self_consistent(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._snapshot(Path(d)/"model")
            self._rebuild_manifest(root,revision="a"*40)
            with self.assertRaises(ValueError):
                validate_hf_snapshot(root,verify_hashes=True)

    def test_rejects_hash_corruption_unsafe_config_and_extra_files(self):
        for issue in ("changed_weight","auto_map","extra"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as d:
                root=self._snapshot(Path(d)/"model")
                if issue=="changed_weight":
                    (root/"model.safetensors").write_bytes(b"changed")
                if issue=="auto_map":
                    config=json.loads((root/"config.json").read_text())
                    config["auto_map"]={"AutoModel":"bad/remote.py"}
                    (root/"config.json").write_text(json.dumps(config))
                    self._rebuild_manifest(root)
                if issue=="extra":
                    (root/"remote.py").write_text("raise Exception('untrusted')")
                    self._rebuild_manifest(root)
                with self.assertRaises(ValueError):
                    validate_hf_snapshot(root,verify_hashes=True)

    def test_backend_uses_repo_owned_pinned_helpers_not_absent_old_modules(self):
        backend=Path("neral_agent_hf.py").read_text(encoding="utf8")
        self.assertIn("from neral_qwen_verified import",backend)
        self.assertIn("validate_hf_snapshot",backend)
        self.assertIn("CHAT_TEMPLATE",backend)
        self.assertNotIn("from model_catalog import validate_hf_snapshot",backend)
        self.assertNotIn("from qwen_local_runtime import CHAT_TEMPLATE",backend)
        helper=Path("neral_qwen_verified.py").read_text(encoding="utf8")
        self.assertNotIn("snapshot_download",helper)
        self.assertNotIn("urllib.request",helper)
        self.assertNotIn("import torch",helper)
        self.assertIn(BASE_REVISION,helper)


if __name__=="__main__":
    unittest.main()
