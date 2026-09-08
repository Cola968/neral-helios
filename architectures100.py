"""Exact parameter contract for the independent Neral-100M Helix family."""
from model100 import HELIX_100, Neral100Config

HELIX_100_EXPECTED_PARAMETERS = 99_783_168


def expected_parameters100(cfg: Neral100Config) -> int:
    d = cfg.d_model
    hd = cfg.head_dim
    kv = cfg.n_kv_heads * hd
    # Per block: Q + K + V + output; Q/K head RMS scales; two residual RMS norms;
    # SwiGLU gate/up/down. Input/output embeddings are tied.
    attention = d * d + d * kv + d * kv + d * d
    qk_norm = 2 * hd
    block_norm = 2 * d
    swiglu = 3 * d * cfg.ffn_hidden
    block = attention + qk_norm + block_norm + swiglu
    return cfg.vocab_size * d + cfg.n_layers * block + d


def validate_helix100_contract() -> dict:
    actual = expected_parameters100(HELIX_100)
    if actual != HELIX_100_EXPECTED_PARAMETERS:
        raise ValueError(f"Helix-100M contract drift: {actual}")
    return {
        "family": "Neral-100M Helix",
        "parameters": actual,
        "millions": actual / 1e6,
        "config": HELIX_100.to_dict(),
        "kv_cache_bf16_4096_bytes": 64 * 1024 * 1024,
        "kv_cache_bf16_8192_bytes": 128 * 1024 * 1024,
        "foreign_weights": False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(validate_helix100_contract(), indent=2))
