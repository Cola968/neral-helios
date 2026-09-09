from __future__ import annotations

"""Neral 600M architecture-search registry.

All candidates are from-scratch dense decoder-only Transformers using Neral's own
16,384-token Byte-BPE vocabulary.  This module contains only analytical contracts:
parameter accounting, approximate FLOP accounting and the locked candidate registry.
It deliberately does not declare a winner before RTX-4070 + matched learning pilots.
"""

from dataclasses import dataclass, asdict
import math

VOCAB_SIZE = 16_384


@dataclass(frozen=True)
class Arch600:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    max_seq_len: int = 8_192
    rope_theta: float = 500_000.0
    attention_pattern: str = "full"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_ratio(self) -> float:
        return self.n_heads / self.n_kv_heads

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"{self.name}: d_model not divisible by heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"{self.name}: Q heads not divisible by KV heads")
        if self.head_dim % 8:
            raise ValueError(f"{self.name}: head_dim must be Tensor-Core friendly")
        if not 24 <= self.n_layers <= 40:
            raise ValueError(f"{self.name}: layer count outside search contract")
        total = self.parameter_breakdown()["total"]
        if not 575_000_000 <= total <= 625_000_000:
            raise ValueError(f"{self.name}: {total:,} params outside 575M-625M")

    def parameter_breakdown(self) -> dict[str, int]:
        # Tied token embedding/output head: vocabulary matrix is counted once.
        d = self.d_model
        hd = self.head_dim
        kv = self.kv_dim
        embedding = VOCAB_SIZE * d
        attention_per_layer = 2 * d * d + 2 * d * kv  # Q + O + K + V
        ffn_per_layer = 3 * d * self.ffn_hidden        # SwiGLU gate/up/down
        norm_per_layer = 2 * d + 2 * hd                # Pre RMSNorms + shared Q/K RMSNorm vectors
        final_norm = d
        per_layer = attention_per_layer + ffn_per_layer + norm_per_layer
        total = embedding + self.n_layers * per_layer + final_norm
        return {
            "embedding": embedding,
            "attention_per_layer": attention_per_layer,
            "ffn_per_layer": ffn_per_layer,
            "norm_per_layer": norm_per_layer,
            "per_layer": per_layer,
            "all_layers": self.n_layers * per_layer,
            "final_norm": final_norm,
            "total": total,
            "trainable": total,
        }

    def analytical_forward_flops_per_token(self, seq_len: int) -> int:
        """Approximate dense forward FLOPs/token.

        Counts multiply-add as 2 FLOPs. Includes QKV/O + MLP linear GEMMs,
        QK^T/AV attention work and tied LM-head projection. Norm/activation/RoPE
        elementwise cost is omitted because it is small relative to GEMMs.
        """
        if seq_len <= 0 or seq_len > self.max_seq_len:
            raise ValueError("invalid sequence length")
        p = self.parameter_breakdown()
        dense_linear = 2 * self.n_layers * (
            p["attention_per_layer"] + p["ffn_per_layer"]
        )
        attention_scores_and_values = self.n_layers * 4 * seq_len * self.d_model
        lm_head = 2 * self.d_model * VOCAB_SIZE
        return dense_linear + attention_scores_and_values + lm_head

    def analytical_train_flops_per_token(self, seq_len: int) -> int:
        # Training backward for matmuls is roughly 2x forward; total ~= 3x forward.
        return 3 * self.analytical_forward_flops_per_token(seq_len)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update({
            "head_dim": self.head_dim,
            "kv_dim": self.kv_dim,
            "gqa_ratio": self.gqa_ratio,
            "parameters": self.parameter_breakdown(),
        })
        return out


# Phase-A analytical search set. 10 serious dense candidates, all within target range.
# Names describe the hypothesis, not a measured ranking.
CANDIDATES: dict[str, Arch600] = {
    # Wide/shallow: large GEMMs, few serial blocks; strongest raw RTX-4070 throughput hypothesis.
    "w24": Arch600("W24", 1536, 24, 24, 6, 3904),
    "w26": Arch600("W26", 1536, 26, 24, 6, 3520),
    "w28": Arch600("W28", 1536, 28, 24, 6, 3200),
    # Mid-width geometries explore more MLP capacity and intermediate depth.
    "b28": Arch600("B28", 1344, 28, 21, 3, 4096),
    "b30": Arch600("B30", 1280, 30, 20, 4, 4032),
    # 4:1 GQA + balanced depth/width: primary pre-hardware sweet-spot hypothesis.
    "b32": Arch600("B32", 1280, 32, 20, 5, 3648),
    # Prime-like deeper/narrower scale-up from the 350M research direction.
    "d36": Arch600("D36", 1152, 36, 18, 3, 3776),
    "d40": Arch600("D40", 1024, 40, 16, 4, 3904),
    # FFN-heavy control: tests knowledge/MLP allocation with aggressive GQA.
    "f27": Arch600("F27", 1152, 27, 18, 2, 5376),
    # Quality-first deep balanced candidate; speed is deliberately secondary.
    "q36": Arch600("Q36", 1280, 36, 20, 5, 3136),
}

# Pre-hardware finalists for Phase C. They cover four distinct scaling regimes.
PHASE_C = ("w26", "b32", "d36", "q36")

# No winner is legal until measured Phase C + Phase D/E evidence exists.
PRE_HARDWARE_SWEET_SPOT = "b32"
PRE_HARDWARE_MAX_QUALITY = "q36"


WEIGHTS = {
    "model_quality_learning_curve": 0.40,
    "quality_per_parameter": 0.20,
    "training_speed": 0.15,
    "vram": 0.10,
    "long_context_efficiency": 0.10,
    "inference_efficiency": 0.05,
}


def normalized_score(metrics: dict[str, float]) -> float:
    """Weighted post-pilot score. Inputs must already be normalized to [0,1], higher=better."""
    missing = set(WEIGHTS) - set(metrics)
    if missing:
        raise ValueError(f"missing score metrics: {sorted(missing)}")
    if any(not 0.0 <= metrics[k] <= 1.0 for k in WEIGHTS):
        raise ValueError("normalized metrics must be in [0,1]")
    return sum(WEIGHTS[k] * metrics[k] for k in WEIGHTS)


def report(seq_len: int = 512) -> list[dict]:
    rows = []
    for key, cfg in CANDIDATES.items():
        cfg.validate()
        p = cfg.parameter_breakdown()
        rows.append({
            "id": key,
            "name": cfg.name,
            "layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "q_heads": cfg.n_heads,
            "kv_heads": cfg.n_kv_heads,
            "head_dim": cfg.head_dim,
            "gqa": cfg.gqa_ratio,
            "ffn": cfg.ffn_hidden,
            **p,
            "forward_flops_per_token": cfg.analytical_forward_flops_per_token(seq_len),
            "train_flops_per_token": cfg.analytical_train_flops_per_token(seq_len),
        })
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps({
        "vocab_size": VOCAB_SIZE,
        "phase_c": PHASE_C,
        "weights": WEIGHTS,
        "rows": report(),
    }, indent=2))
