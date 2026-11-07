"""NERAL v5 opt-in Helios model selection; no GPU, no weights or network."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_auto_api import resolve_local_adapter_paths, register_autonomous_agent_api
from flask import Flask


class NeralV5ChannelTests(unittest.TestCase):
    def test_default_keeps_existing_rank32_legacy_channel(self):
        home=Path("/home/local")
        base,adapter,explicit=resolve_local_adapter_paths(home,{})
        self.assertFalse(explicit)
        self.assertEqual(adapter,home/"NERAL-QLoRA-Research"/"actions-r32-42078722575-1"/"adapter")
        self.assertEqual(base,home/".cache"/"neral"/"qwen25-smoke"/
                         "c03e6d358207e414f1eca0bb1891e29f1db0e242"/
                         "installed_models"/"hf-qwen25-coder-7b")

    def test_explicit_v5_uses_exact_saved_trained_not_promoted_checkpoint(self):
        home=Path("/home/local")
        base,adapter,explicit=resolve_local_adapter_paths(home,{
            "NERAL_AGENT_ADAPTER_CHANNEL":"v5",
        })
        self.assertTrue(explicit)
        self.assertEqual(adapter,home/"NERAL-QLoRA-Research"/
                         "v5-r16-480-42378021782-1"/"adapter")
        self.assertEqual(base.name,"hf-qwen25-coder-7b")

    def test_explicit_paths_override_derived_paths(self):
        home=Path("/home/local")
        base,adapter,explicit=resolve_local_adapter_paths(home,{
            "NERAL_AGENT_ADAPTER_CHANNEL":"v5",
            "NERAL_AGENT_BASE_PATH":"/local/models/base",
            "NERAL_AGENT_ADAPTER_PATH":"/local/adapters/reviewed",
        })
        self.assertEqual(base,Path("/local/models/base"))
        self.assertEqual(adapter,Path("/local/adapters/reviewed"))
        self.assertTrue(explicit)

    def test_rejects_unknown_channel_and_empty_override(self):
        for channel in ("unreviewed","V5", "", "cloud", "../v5"):
            with self.subTest(channel=channel):
                with self.assertRaises(ValueError):
                    resolve_local_adapter_paths(Path("/home/local"),{
                        "NERAL_AGENT_ADAPTER_CHANNEL":channel,
                    })
        with self.assertRaises(ValueError):
            resolve_local_adapter_paths(Path("/home/local"),{
                "NERAL_AGENT_ADAPTER_CHANNEL":"v5",
                "NERAL_AGENT_BASE_PATH":" ",
            })

    def test_v5_opt_in_never_silently_falls_back_to_ollama(self):
        with tempfile.TemporaryDirectory() as temp:
            env={"NERAL_AGENT_BACKEND":"auto","NERAL_AGENT_ADAPTER_CHANNEL":"v5",
                 "NERAL_AGENT_BASE_PATH":str(Path(temp)/"missing-base"),
                 "NERAL_AGENT_ADAPTER_PATH":str(Path(temp)/"missing-adapter")}
            with patch.dict(os.environ,env,clear=True):
                app=Flask("neral-selection-missing")
                with self.assertRaises((ValueError,OSError)):
                    register_autonomous_agent_api(app,Path(temp))

    def test_manual_hf_v5_channel_avoids_fallback_without_explicit_paths(self):
        home=Path("/home/local")
        base,adapter,selected=resolve_local_adapter_paths(home,{
            "NERAL_AGENT_ADAPTER_CHANNEL":"v5",
            "NERAL_AGENT_BACKEND":"hf_adapter",
        })
        self.assertTrue(selected)
        self.assertTrue(str(adapter).endswith("v5-r16-480-42378021782-1/adapter"))
        self.assertTrue(base.parts[-1]=="hf-qwen25-coder-7b")


if __name__=="__main__":
    unittest.main()
