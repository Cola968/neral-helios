"""Portable CPU tests: never serve v6/v61 research weights; no CUDA needed."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from neral_hf_identity import QUARANTINED_ADAPTER_SHA256 as HF_BLOCKED
from mini.neral_mini import QUARANTINED_ADAPTER_SHA256 as MINI_BLOCKED

V6 = "8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd"
V61 = "33e939f6063d4215807f593816a057e88d8938b30615c6000d6a0bf8c7965279"
V5 = "2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5"


def test_hf_and_mini_same_quarantine_while_v5_remains_eligible():
    assert HF_BLOCKED == MINI_BLOCKED
    assert {V6,V61} <= HF_BLOCKED
    assert V5 not in HF_BLOCKED


def test_local_agent_rejects_v61_hash_before_cuda_loading():
    from neral_agent_hf import HFAdapterLocalBackend
    from neral_agent_autonomy import AgentModelError
    with TemporaryDirectory() as t:
        root=Path(t)
        base=root/"base"
        adapter=root/"adapter"
        base.mkdir()
        adapter.mkdir()
        (adapter/"adapter_model.safetensors").write_bytes(b"placeholder-not-real")
        (adapter/"adapter_config.json").write_text(
            json.dumps({"r":16,"peft_type":"LORA"}),encoding="utf8")
        with patch("neral_agent_hf.hashlib.file_digest",
                   return_value=Mock(hexdigest=lambda:V61)):
            try:
                HFAdapterLocalBackend(base,adapter)
            except AgentModelError as exc:
                assert "gesperrt" in str(exc)
            else:
                raise AssertionError("v6.1 invalid candidate was incorrectly accepted")
