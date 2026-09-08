"""Versioned checkpoint, initialization, and resume contracts for Neral-100M Helix."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from architectures100_variants import EXPECTED, VARIANTS
from model100 import Neral100M

FORMAT_VERSION = 2
TOKENIZER_CONTRACT = {"kind": "neral-byte", "vocab_size": 260, "bos": 1, "eos": 2, "byte_offset": 4}


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tokenizer_contract_sha256() -> str:
    return stable_hash(TOKENIZER_CONTRACT)


def make_training_contract(
    variant: str,
    *,
    seq: int,
    microbatch: int,
    grad_accum: int,
    lr: float,
    min_lr: float,
    warmup_fraction: float,
    weight_decay: float = .1,
    beta1: float = .9,
    beta2: float = .95,
    grad_clip: float = 1.0,
) -> dict:
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    if min(seq, microbatch, grad_accum) <= 0 or lr <= 0 or min_lr < 0 or not 0 <= warmup_fraction < 1:
        raise ValueError("invalid training contract")
    if min_lr > lr:
        raise ValueError("min_lr cannot exceed lr")
    return {
        "version": 1,
        "optimizer": "AdamW",
        "variant": variant,
        "requested_seq": int(seq),
        "requested_microbatch": int(microbatch),
        "grad_accum": int(grad_accum),
        "peak_lr": float(lr),
        "min_lr": float(min_lr),
        "warmup_fraction": float(warmup_fraction),
        "betas": [float(beta1), float(beta2)],
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
        "precision": "bf16-autocast",
        "gradient_checkpointing": True,
        "foreign_weights": False,
    }


def init_audit(variant: str, seed: int = 100968) -> dict:
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    torch.manual_seed(seed)
    model = Neral100M(VARIANTS[variant])
    state = model.state_dict()
    finite = all((not torch.is_floating_point(v)) or torch.isfinite(v).all().item() for v in state.values())
    tied = model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params != EXPECTED[variant] or not finite or not tied:
        raise ValueError("invalid Helix initialization")
    return {
        "variant": variant,
        "parameters": params,
        "finite": finite,
        "tied_embeddings": tied,
        "config_sha256": stable_hash(VARIANTS[variant].to_dict()),
        "tokenizer_contract_sha256": tokenizer_contract_sha256(),
        "seed": seed,
    }


def checkpoint_payload(
    model: Neral100M,
    variant: str,
    *,
    tokens: int,
    step: int,
    optimizer_state: dict | None = None,
    training_contract: dict | None = None,
    rng_state: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    if variant not in VARIANTS or model.cfg != VARIANTS[variant]:
        raise ValueError("model/config mismatch")
    if min(tokens, step) < 0:
        raise ValueError("nonnegative counters required")
    config = model.cfg.to_dict()
    payload = {
        "format": "neral-helix-checkpoint",
        "format_version": FORMAT_VERSION,
        "architecture_family": "Neral-100M Helix",
        "variant": variant,
        "config": config,
        "config_sha256": stable_hash(config),
        "tokenizer_contract": TOKENIZER_CONTRACT,
        "tokenizer_contract_sha256": tokenizer_contract_sha256(),
        "parameters": EXPECTED[variant],
        "model": model.state_dict(),
        "tokens": int(tokens),
        "step": int(step),
        "provenance": provenance or {},
        "foreign_weights": False,
    }
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    if training_contract is not None:
        payload["training_contract"] = training_contract
        payload["training_contract_sha256"] = stable_hash(training_contract)
    if rng_state is not None:
        payload["rng_state"] = rng_state
    return payload


def _validate_state_shapes(variant: str, state: dict) -> None:
    if not isinstance(state, dict) or not state:
        raise ValueError("invalid state")
    reference = Neral100M(VARIANTS[variant]).state_dict()
    if set(state) != set(reference):
        missing = sorted(set(reference) - set(state))[:5]
        extra = sorted(set(state) - set(reference))[:5]
        raise ValueError(f"state key mismatch missing={missing} extra={extra}")
    for key, tensor in state.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"non-tensor state: {key}")
        if tuple(tensor.shape) != tuple(reference[key].shape):
            raise ValueError(f"shape mismatch: {key}")
        if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and not torch.isfinite(tensor).all().item():
            raise ValueError(f"non-finite state: {key}")


def validate_checkpoint(payload: dict, require_optimizer: bool = False, require_training_contract: bool = False) -> dict:
    if payload.get("format") != "neral-helix-checkpoint" or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported format")
    variant = payload.get("variant")
    if variant not in VARIANTS or payload.get("parameters") != EXPECTED[variant]:
        raise ValueError("parameter contract mismatch")
    if payload.get("foreign_weights") is not False:
        raise ValueError("foreign weights flag")
    expected_config = VARIANTS[variant].to_dict()
    if payload.get("config") != expected_config:
        raise ValueError("config body mismatch")
    if payload.get("config_sha256") != stable_hash(expected_config):
        raise ValueError("config hash mismatch")
    if payload.get("tokenizer_contract") != TOKENIZER_CONTRACT:
        raise ValueError("tokenizer contract body mismatch")
    if payload.get("tokenizer_contract_sha256") != tokenizer_contract_sha256():
        raise ValueError("tokenizer contract hash mismatch")
    if int(payload.get("tokens", -1)) < 0 or int(payload.get("step", -1)) < 0:
        raise ValueError("invalid counters")
    if require_optimizer and "optimizer" not in payload:
        raise ValueError("optimizer required")
    contract = payload.get("training_contract")
    if require_training_contract and not isinstance(contract, dict):
        raise ValueError("training contract required")
    if contract is not None:
        if contract.get("variant") != variant or contract.get("foreign_weights") is not False:
            raise ValueError("training contract mismatch")
        if payload.get("training_contract_sha256") != stable_hash(contract):
            raise ValueError("training contract hash mismatch")
    _validate_state_shapes(variant, payload.get("model"))
    return {
        "variant": variant,
        "parameters": EXPECTED[variant],
        "tokens": int(payload["tokens"]),
        "step": int(payload["step"]),
        "valid": True,
    }


def experimental_50m_migration_plan(source_config: dict, target_variant: str) -> dict:
    if target_variant not in VARIANTS:
        raise ValueError("unknown target")
    target = VARIANTS[target_variant]
    return {
        "status": "experimental_only",
        "production_pretraining": "from_scratch",
        "target_variant": target_variant,
        "target_parameters": EXPECTED[target_variant],
        "source_d_model": source_config.get("d_model"),
        "target_d_model": target.d_model,
        "allowed_mapping": ["overlapping tokenizer rows", "exact-shape tensors only"],
        "forbidden_claim": "function-preserving or equivalent initialization",
        "requires_separate_ablation": True,
    }
