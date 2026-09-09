from __future__ import annotations

"""ORION A5: conservative throughput candidate for the ~100M Neral generation.

A5 keeps the proven A3 tensor-core-friendly geometry (d_model=768, head_dim=64,
12 query heads, 2 KV heads) but shifts capacity from depth into the SwiGLU FFN.
The goal is lower launch/KV overhead without taking the higher quality risk of MQA.

This file defines an experimental candidate only. It is NOT promoted until it wins
real RTX-4070 throughput/VRAM probes and frozen quality gates after training.
"""

from orion100 import OrionConfig, NeralOrion, expected_parameters, architecture_report


# 99,900,672 parameters. Relative to A3:
# - layers: 14 -> 12 (-14.29%)
# - linear GEMM calls/forward: 56 -> 48 (-14.29%)
# - BF16 KV cache at equal context: -14.29%
# - FFN hidden: 2496 -> 3008 (+20.51%)
# Attention geometry stays unchanged (12Q/2KV, head_dim=64).
ORION_A5 = OrionConfig(
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=2,
    ffn_hidden=3008,
    max_seq_len=8192,
    rope_theta=100_000.0,
)


def a5_report() -> dict:
    report = architecture_report(ORION_A5)
    report.update(
        candidate="ORION_A5",
        design="balanced-launch",
        expected_parameters_exact=expected_parameters(ORION_A5),
        promotion_status="experimental",
        rationale={
            "keep_head_dim_64": True,
            "keep_gqa_12q_2kv": True,
            "reduce_layer_launch_depth_vs_a3_pct": 14.2857142857,
            "reduce_kv_cache_vs_a3_pct": 14.2857142857,
            "increase_ffn_hidden_vs_a3_pct": 20.5128205128,
        },
    )
    return report


if __name__ == "__main__":
    import json

    model = NeralOrion(ORION_A5)
    payload = a5_report()
    payload["runtime_parameter_count"] = sum(p.numel() for p in model.parameters())
    print(json.dumps(payload, indent=2))
