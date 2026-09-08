"""Hardware-search candidates for Neral-100M Helix.

A1 follows the strongest deep/thin small-model evidence. A2 keeps the same modern Helix
building blocks but uses fewer, wider layers to reduce per-token kernel-launch depth on the
RTX 4070 while preserving roughly 100M trainable parameters.
"""
from model100 import HELIX_100, Neral100Config
from architectures100 import expected_parameters100

HELIX_A1 = HELIX_100
HELIX_A2 = Neral100Config(
    vocab_size=260,
    d_model=640,
    n_layers=24,
    n_heads=10,
    n_kv_heads=2,
    ffn_hidden=1664,
    max_seq_len=8192,
    rope_theta=100_000.0,
    dropout=0.0,
    rms_eps=1e-5,
    qk_rms_eps=1e-6,
    initializer_range=0.02,
)

EXPECTED = {
    "a1": 99_783_168,
    "a2": 100_470_912,
}

VARIANTS = {"a1": HELIX_A1, "a2": HELIX_A2}


def report(name: str) -> dict:
    if name not in VARIANTS:
        raise ValueError(f"unknown Helix variant: {name}")
    cfg = VARIANTS[name]
    params = expected_parameters100(cfg)
    if params != EXPECTED[name]:
        raise ValueError(f"Helix {name} parameter drift: {params}")
    bytes_4096 = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 4096 * 2
    bytes_8192 = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 8192 * 2
    return {
        "variant": name,
        "parameters": params,
        "millions": params / 1e6,
        "config": cfg.to_dict(),
        "kv_cache_bf16_4096_bytes": bytes_4096,
        "kv_cache_bf16_8192_bytes": bytes_8192,
    }


if __name__ == "__main__":
    import json
    print(json.dumps({name: report(name) for name in VARIANTS}, indent=2))
