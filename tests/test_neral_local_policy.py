"""Local-only inference must be explicit, no benchmark claims from names."""
import pytest
from scripts.neral_local_policy import OFFLINE_ENV, enforce_local_inference_environment, inference_provenance


def test_offline_env_overrides_network_telemetry():
    env = {'HF_HUB_OFFLINE': '0', 'WANDB_MODE': 'online'}
    enforce_local_inference_environment(env)
    assert all(env[k] == v for k, v in OFFLINE_ENV.items())


def test_model_identity_does_not_claim_independent_model():
    base = inference_provenance(adapter_loaded=False)
    assert base['neral_adapter_loaded'] is False
    assert 'foundation only' in base['model_display_name']
    assert base['independent_from_qwen_weights'] is False
    with pytest.raises(ValueError):
        inference_provenance(adapter_loaded=False, adapter_sha256='a'*64)
    with pytest.raises(ValueError):
        inference_provenance(adapter_loaded=True)
    adapter = inference_provenance(adapter_loaded=True, adapter_sha256='a'*64)
    assert adapter['neral_adapter_sha256'] == 'a'*64
    assert adapter['model_display_name'].startswith('NERAL')
    assert adapter['inference_location'] == 'local_device'

def test_qwen_pair_comparison_uses_forced_offline_and_local_weights():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / 'scripts' / 'neral_r32_compare.py').read_text()
    assert 'enforce_local_inference_environment()' in src
    assert src.index('enforce_local_inference_environment()') < src.index('from transformers import Qwen2TokenizerFast')
    assert 'local_files_only=True' in src
    assert 'token=False' in src
    assert 'PeftModel.from_pretrained' in src
    assert "adapter_status='experimental_loaded_not_promoted'" in src
