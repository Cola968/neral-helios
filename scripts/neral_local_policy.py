"""NERAL offline inference policy and honest model provenance.

Disables Hugging Face downloads and WANDB for reviewed local inference.
Does not act as an operating-system firewall: internet-enabled plugins
must have their own explicit opt-in and access controls.
"""
from __future__ import annotations
import os
import re

OFFLINE_ENV = {
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_DATASETS_OFFLINE': '1',
    'WANDB_DISABLED': 'true',
    'WANDB_MODE': 'disabled',
}
FOUNDATION = 'Qwen/Qwen2.5-Coder-7B-Instruct'


def enforce_local_inference_environment(environ=None):
    env = os.environ if environ is None else environ
    for key, value in OFFLINE_ENV.items():
        env[key] = value


def inference_provenance(*, adapter_loaded: bool, adapter_sha256: str | None = None) -> dict:
    """Only label a model NERAL when a verified adapter was actually loaded."""
    if type(adapter_loaded) is not bool:
        raise ValueError('adapter_loaded must be a boolean')
    if adapter_loaded:
        if not isinstance(adapter_sha256, str) or not re.fullmatch(r'[a-f0-9]{64}', adapter_sha256):
            raise ValueError('Adapter hash required to identify NERAL model')
    elif adapter_sha256 is not None:
        raise ValueError('Cannot claim an adapter for foundation-only inference')
    return {
        'model_display_name': 'NERAL (Qwen foundation + verified adapter)' if adapter_loaded else 'Qwen2.5-Coder foundation only',
        'foundation_model': FOUNDATION,
        'foundation_role': 'locally_executed_frozen_base_weights',
        'neral_adapter_loaded': adapter_loaded,
        'neral_adapter_sha256': adapter_sha256,
        'inference_location': 'local_device',
        'data_sent_to_qwen_service': False,
        'independent_from_qwen_weights': False,
    }
