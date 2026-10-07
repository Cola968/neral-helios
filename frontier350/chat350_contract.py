from __future__ import annotations

"""Fail-closed compatibility contract for Arctus/Neral 350M chat post-training.

This module intentionally has no torch dependency so it can be validated on GitHub-hosted
runners while the local RTX machine is offline.
"""

ARCHITECTURE = "frontier350-v1"
EXPECTED_PARAMS = 349_825_152
EXPECTED_TOKENIZER_SHA256 = "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
EXPECTED_SPECIALS = {
    "<|bos|>": 0,
    "<|eos|>": 1,
    "<|sep|>": 2,
    "<|pad|>": 3,
}
EXPECTED_CONFIG = {
    "vocab_size": 16_384,
    "d_model": 1_152,
    "n_layers": 24,
    "n_heads": 18,
    "n_kv_heads": 2,
    "ffn_hidden": 3_136,
    "max_seq_len": 8_192,
    "rope_theta": 250_000.0,
    "rms_eps": 1e-5,
    "qk_rms_eps": 1e-6,
    "dropout": 0.0,
    "initializer_range": 0.02,
}


def validate_tokenizer_lock(lock: dict) -> None:
    if lock.get("architecture") != ARCHITECTURE:
        raise ValueError("tokenizer lock architecture mismatch")
    if int(lock.get("vocab_size", -1)) != EXPECTED_CONFIG["vocab_size"]:
        raise ValueError("tokenizer vocabulary mismatch")
    if lock.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("tokenizer digest mismatch")
    if lock.get("special_token_ids") != EXPECTED_SPECIALS:
        raise ValueError("special-token mapping mismatch")
    if lock.get("foreign_vocab_or_merges_loaded") is not False:
        raise ValueError("foreign tokenizer components are not allowed")


def validate_checkpoint_metadata(meta: dict, *, min_processed_tokens: int = 1_000_000_000) -> None:
    """Validate metadata before loading hundreds of millions of weights.

    Chat post-training deliberately starts a fresh optimizer, so an optimizer state is not
    required. The model weights, architecture identity and tokenizer binding are required.
    """
    if meta.get("config") != EXPECTED_CONFIG:
        raise ValueError("checkpoint architecture/config mismatch")
    if meta.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("checkpoint/tokenizer digest mismatch")
    processed = int(meta.get("processed_tokens", meta.get("tokens", 0)))
    if processed < min_processed_tokens:
        raise ValueError(f"checkpoint is too early for chat recovery: {processed:,} tokens")
    if "model" not in meta:
        raise ValueError("checkpoint has no model state")


def release_identity() -> dict:
    return {
        "architecture": ARCHITECTURE,
        "parameters": EXPECTED_PARAMS,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "special_token_ids": dict(EXPECTED_SPECIALS),
        "config": dict(EXPECTED_CONFIG),
        "posttraining_optimizer_policy": "fresh_optimizer_from_immutable_foundation_checkpoint",
    }
