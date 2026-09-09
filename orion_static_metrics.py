from __future__ import annotations

"""Pure-CPU static architecture accounting for Neral/Helix/ORION.

Nothing in this module measures CUDA performance. Exact counts are algebraic architecture
contracts; FLOPs are analytical estimates; training-VRAM values are deliberately coarse planning
ranges with their assumptions embedded in the report.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

MIB = 1024 ** 2


@dataclass(frozen=True)
class StaticSpec:
    name: str
    family: str
    parameters: int
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden: int
    max_seq_len: int
    fused_qkv: bool
    fused_gate_up: bool
    qk_norm: bool

    def validate(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(f"{self.name}: inconsistent head geometry")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"{self.name}: invalid GQA grouping")
        if self.head_dim % 2:
            raise ValueError(f"{self.name}: RoPE head_dim must be even")
        if min(self.parameters, self.n_layers, self.ffn_hidden, self.max_seq_len) <= 0:
            raise ValueError(f"{self.name}: positive dimensions required")


SPECS = {
    "50m_legacy": StaticSpec("Neral 50M Legacy", "Neral-50M", 50_012_976, 260, 624, 8, 12, 12, 52, 2496, 4096, True, False, False),
    "50m_fused": StaticSpec("Neral 50M Fused", "Neral-50M", 50_012_976, 260, 624, 8, 12, 12, 52, 2496, 4096, True, True, False),
    "helix_a1": StaticSpec("Helix A1", "Neral-100M Helix", 99_783_168, 260, 512, 32, 16, 4, 32, 1600, 8192, False, False, True),
    "helix_a2": StaticSpec("Helix A2", "Neral-100M Helix", 100_470_912, 260, 640, 24, 10, 2, 64, 1664, 8192, False, False, True),
    "orion_a3": StaticSpec("ORION A3", "Neral-100M ORION", 100_002_304, 260, 768, 14, 12, 2, 64, 2496, 8192, True, True, True),
    "orion_a4": StaticSpec("ORION A4", "Neral-100M ORION", 100_103_936, 260, 768, 16, 12, 2, 64, 2112, 8192, True, True, True),
}


def parameter_partition(spec: StaticSpec) -> dict:
    spec.validate()
    d, h, kv, hd, f = spec.d_model, spec.n_heads, spec.n_kv_heads, spec.head_dim, spec.ffn_hidden
    if spec.family == "Neral-50M":
        attention_linear = 4 * d * d
        qk_norm = 0
    else:
        kv_dim = kv * hd
        # Q + K + V + O. Fusing projections changes launches, not represented parameters.
        attention_linear = d * d + 2 * d * kv_dim + d * d
        qk_norm = 2 * hd if spec.qk_norm else 0
    attention = attention_linear + qk_norm
    ffn = 3 * d * f
    block_norms = 2 * d
    per_layer = attention + ffn + block_norms
    embeddings = spec.vocab_size * d
    final_norm = d
    total = embeddings + spec.n_layers * per_layer + final_norm
    if total != spec.parameters:
        raise ValueError(f"{spec.name}: parameter drift {total:,} != {spec.parameters:,}")
    return {
        "total": total,
        "embeddings_tied": embeddings,
        "per_layer": per_layer,
        "attention": attention,
        "attention_linear": attention_linear,
        "ffn": ffn,
        "block_norms": block_norms,
        "final_norm": final_norm,
        "attention_fraction_per_layer": attention / per_layer,
        "ffn_fraction_per_layer": ffn / per_layer,
    }


def linear_gemm_calls(spec: StaticSpec) -> dict:
    qkv = 1 if spec.fused_qkv else 3
    gate_up = 1 if spec.fused_gate_up else 2
    per_block = qkv + 1 + gate_up + 1  # QKV + O + gate/up + down
    return {"per_block": per_block, "per_forward": per_block * spec.n_layers}


def kv_cache_bytes(spec: StaticSpec, seq: int, batch: int = 1, bytes_per_value: int = 2) -> int:
    if not 0 < seq <= spec.max_seq_len:
        raise ValueError(f"{spec.name}: seq outside context")
    return batch * spec.n_layers * 2 * spec.n_kv_heads * seq * spec.head_dim * bytes_per_value


def theoretical_forward_flops(spec: StaticSpec, seq: int, batch: int = 1) -> int:
    """Approximate dense forward FLOPs, excluding norms/activations/embedding lookup.

    Linear maps count one multiply-add as 2 FLOPs. Causal SDPA uses the common dense upper-bound
    4*B*L*T^2*d for QK^T and AV. Flash/causal kernels reduce realized work/memory, so this is a
    comparison estimate rather than a profiler measurement.
    """
    if not 0 < seq <= spec.max_seq_len:
        raise ValueError("sequence outside context")
    p = parameter_partition(spec)
    linear_per_layer = p["attention_linear"] + p["ffn"]
    linear = 2 * batch * seq * spec.n_layers * linear_per_layer
    dense_attention = 4 * batch * spec.n_layers * seq * seq * spec.d_model
    return linear + dense_attention


def activation_lower_bound_bytes(spec: StaticSpec, seq: int, batch: int = 1) -> int:
    """BF16 residual-stream lower bound, not a full training allocation estimate."""
    return batch * seq * spec.d_model * (spec.n_layers + 1) * 2


def training_state_range_bytes(spec: StaticSpec) -> tuple[int, int]:
    """Planning range for weights+gradients+Adam state, excluding activations/runtime.

    Lower: BF16 weights + BF16 grads + two BF16 Adam moments = 8 B/parameter.
    Upper: BF16 weights + FP32 grads + FP32 moments + optional FP32 master = 18 B/parameter.
    Actual PyTorch layout must be measured on the RTX runner.
    """
    return spec.parameters * 8, spec.parameters * 18


def planning_training_vram_range_bytes(spec: StaticSpec, seq: int, batch: int = 1) -> tuple[int, int]:
    """Explicitly heuristic VRAM planning band; never report as measured.

    It adds a fixed 512 MiB runtime allowance and 4x..20x the residual lower bound to the training
    state range. This intentionally broad range exists only to reject obviously impossible matrix
    points before the real RTX-4070 benchmark.
    """
    lo_state, hi_state = training_state_range_bytes(spec)
    residual = activation_lower_bound_bytes(spec, seq, batch)
    runtime = 512 * MIB
    return lo_state + 4 * residual + runtime, hi_state + 20 * residual + runtime


def report_spec(spec: StaticSpec) -> dict:
    p = parameter_partition(spec)
    gems = linear_gemm_calls(spec)
    seqs = [s for s in (256, 512, 1024, 2048, 4096, 8192) if s <= spec.max_seq_len]
    rows = {}
    for seq in seqs:
        lo, hi = planning_training_vram_range_bytes(spec, seq)
        rows[str(seq)] = {
            "kv_cache_bf16_bytes": kv_cache_bytes(spec, seq),
            "theoretical_forward_flops_b1": theoretical_forward_flops(spec, seq),
            "activation_residual_lower_bound_bytes_b1": activation_lower_bound_bytes(spec, seq),
            "planning_training_vram_range_bytes_b1": [lo, hi],
        }
    state_lo, state_hi = training_state_range_bytes(spec)
    return {
        "spec": asdict(spec),
        "parameter_partition": p,
        "linear_gemm_calls": gems,
        "bf16_weight_bytes": spec.parameters * 2,
        "training_state_range_bytes_excluding_activations": [state_lo, state_hi],
        "kv_cache_bytes_per_parameter_at_4096": kv_cache_bytes(spec, min(4096, spec.max_seq_len)) / spec.parameters,
        "sequence_matrix": rows,
        "evidence_class": {
            "parameter_and_kv_counts": "calculated-exact",
            "gemm_calls": "structural-calculated",
            "flops": "theoretical-estimate",
            "activation_lower_bound": "calculated-lower-bound",
            "training_vram_range": "planning-heuristic-not-measured",
        },
    }


def build_report() -> dict:
    data = {name: report_spec(spec) for name, spec in SPECS.items()}
    a2 = data["helix_a2"]["linear_gemm_calls"]["per_forward"]
    a3 = data["orion_a3"]["linear_gemm_calls"]["per_forward"]
    return {
        "schema": 1,
        "gpu_measurements_present": False,
        "quality_scores_for_untrained_orion_present": False,
        "models": data,
        "comparisons": {
            "orion_a3_gemm_reduction_vs_helix_a2": 1 - a3 / a2,
            "orion_a3_parameters_vs_50m": SPECS["orion_a3"].parameters / SPECS["50m_legacy"].parameters,
            "orion_a4_parameters_vs_50m": SPECS["orion_a4"].parameters / SPECS["50m_legacy"].parameters,
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# ORION static CPU architecture report",
        "",
        "> No CUDA measurement is contained here. Parameter/KV counts are exact calculations; FLOPs and VRAM planning bands are theoretical.",
        "",
        "| Model | Params | Layers | d_model | Q/KV | head | FFN | GEMMs/fwd | BF16 weights MiB | KV@4K MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in SPECS:
        m = report["models"][key]
        s = m["spec"]
        kv = m["sequence_matrix"]["4096"]["kv_cache_bf16_bytes"] / MIB
        lines.append(
            f"| {s['name']} | {s['parameters']:,} | {s['n_layers']} | {s['d_model']} | {s['n_heads']}/{s['n_kv_heads']} | {s['head_dim']} | {s['ffn_hidden']} | {m['linear_gemm_calls']['per_forward']} | {m['bf16_weight_bytes']/MIB:.2f} | {kv:.2f} |"
        )
    lines += [
        "",
        "## Promotion rule",
        "",
        "Static efficiency may select candidates for measurement, but can never promote ORION. Promotion requires the reproducible RTX-4070 benchmark plus frozen quality gates after real training.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    args = ap.parse_args()
    report = build_report()
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf8")
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(report), encoding="utf8")
    if not args.json and not args.markdown:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
