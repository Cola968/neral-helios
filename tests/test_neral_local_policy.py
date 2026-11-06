"""Local-only inference must be explicit, no benchmark claims from names."""
import pytest
from neral_local_policy import OFFLINE_ENV, enforce_local_inference_environment, inference_provenance


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
