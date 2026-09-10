from __future__ import annotations

"""Analytical architecture contract for the Neral Nova 1B research model.

Nova is a from-scratch Neral model. It does not import or reuse foreign weights,
checkpoints, vocabularies, or tokenizer merges. The default research candidate mixes
Mamba-3 sequence mixers with periodic exact full-attention anchors and a small
Neral-specific Adaptive Memory Weave (AMW) bridge.

The Mamba-3 parameter formula below is pinned to the upstream implementation revision
recorded in UPSTREAM_MAMBA_REVISION. An instantiated model must still verify its real
parameter count before any pilot is accepted.
"""

from dataclasses import asdict, dataclass, replace

VOCAB_SIZE = 16_384
UPSTREAM_MAMBA_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
UPSTREAM_MAMBA_LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class Nova1BConfig:
    name: str = "nova1b-m3-amw-rope"
    d_model: int = 1536
    n_layers: int = 32
    ffn_hidden: int = 3648
    head_dim: int = 64
    n_heads: int = 24
    n_kv_heads: int = 6
    max_seq_len: int = 16_384
    rope_theta: float = 500_000.0

    # Hybrid schedule: M, M, M, A repeated eight times.
    mamba_per_cycle: int = 3
    attention_per_cycle: int = 1
    cycles: int = 8

    # Mamba-3 MIMO research settings.
    mamba_state: int = 128
    mamba_expand: int = 2
    mamba_groups: int = 1
    mamba_mimo_rank: int = 4
    mamba_rope_fraction: float = 0.5
    mamba_outproj_norm: bool = False

    # Neral-specific experimental bridge.
    amw_rank: int = 128
    use_amw: bool = True

    # Full-attention anchor position mode. "nope" is an ablation, not a winner.
    anchor_position: str = "rope"
    gated_attention: bool = True

    def validate(self) -> None:
        if self.n_layers != self.cycles * (self.mamba_per_cycle + self.attention_per_cycle):
            raise ValueError("hybrid schedule does not equal n_layers")
        if self.attention_per_cycle != 1:
            raise ValueError("Nova v1 expects one exact-attention anchor per cycle")
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("GQA requires n_heads divisible by n_kv_heads")
        if self.d_model * self.mamba_expand % self.head_dim:
            raise ValueError("Mamba inner width must divide into head_dim")
        if self.anchor_position not in {"rope", "nope"}:
            raise ValueError("anchor_position must be rope or nope")
        if self.mamba_rope_fraction not in {0.5, 1.0}:
            raise ValueError("unsupported Mamba-3 RoPE fraction")
        if self.amw_rank <= 0 or self.amw_rank > self.d_model:
            raise ValueError("invalid AMW rank")

    @property
    def n_mamba(self) -> int:
        return self.cycles * self.mamba_per_cycle

    @property
    def n_attention(self) -> int:
        return self.cycles * self.attention_per_cycle

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_ratio(self) -> int:
        return self.n_heads // self.n_kv_heads

    def schedule(self) -> tuple[str, ...]:
        unit = ("mamba3",) * self.mamba_per_cycle + ("attention",)
        return unit * self.cycles

    def mamba3_mixer_params(self) -> int:
        """Exact formula for pinned upstream Mamba3 with MIMO rank 4.

        Includes in_proj/out_proj, dt/B/C/D parameters, B/C norm weights and MIMO
        x/z/o factors. It assumes is_outproj_norm=False, bias-free projections and
        the upstream shapes at UPSTREAM_MAMBA_REVISION.
        """
        d = self.d_model
        inner = self.mamba_expand * d
        nheads = inner // self.head_dim
        state = self.mamba_state
        rank = self.mamba_mimo_rank
        groups = self.mamba_groups
        split = int(state * self.mamba_rope_fraction)
        if split % 2:
            split -= 1
        rope_angles = split // 2
        in_width = (
            2 * inner
            + 2 * state * groups * rank
            + 3 * nheads
            + rope_angles
        )
        in_proj = d * in_width
        out_proj = inner * d
        dt_bias = nheads
        bc_bias = 2 * nheads * rank * state
        bc_norm = 2 * state
        mimo_factors = 3 * nheads * rank * self.head_dim
        d_skip = nheads
        out_norm = inner if self.mamba_outproj_norm else 0
        return in_proj + out_proj + dt_bias + bc_bias + bc_norm + mimo_factors + d_skip + out_norm

    def swiglu_params(self) -> int:
        return 3 * self.d_model * self.ffn_hidden

    def attention_mixer_params(self) -> int:
        d = self.d_model
        # packed Q/K/V plus output projection
        projections = d * (d + 2 * self.kv_dim) + d * d
        # Q/K head-dimension RMSNorm vectors
        qk_norm = 2 * self.head_dim
        # Dynamic per-token, per-query-head sigmoid gate after attention.
        gate = d * self.n_heads if self.gated_attention else 0
        return projections + qk_norm + gate

    def block_norm_params(self) -> int:
        return 2 * self.d_model

    def amw_params(self) -> int:
        if not self.use_amw:
            return 0
        # bias-free d->rank->d bridge + per-channel injection gate
        return 2 * self.d_model * self.amw_rank + self.d_model

    def parameter_breakdown(self) -> dict[str, int]:
        self.validate()
        embedding = VOCAB_SIZE * self.d_model  # tied LM head counted once
        mamba_block = self.mamba3_mixer_params() + self.swiglu_params() + self.block_norm_params()
        attention_block = self.attention_mixer_params() + self.swiglu_params() + self.block_norm_params()
        amw_total = self.cycles * self.amw_params()
        final_norm = self.d_model
        total = (
            embedding
            + self.n_mamba * mamba_block
            + self.n_attention * attention_block
            + amw_total
            + final_norm
        )
        return {
            "tied_embedding": embedding,
            "mamba3_mixer_per_block": self.mamba3_mixer_params(),
            "swiglu_per_block": self.swiglu_params(),
            "mamba_block": mamba_block,
            "attention_mixer_per_block": self.attention_mixer_params(),
            "attention_block": attention_block,
            "amw_per_cycle": self.amw_params(),
            "amw_total": amw_total,
            "final_norm": final_norm,
            "total": total,
        }

    def to_dict(self) -> dict:
        return asdict(self)


NOVA_1B = Nova1BConfig()
NOVA_1B_NO_AMW = replace(NOVA_1B, name="nova1b-m3-nobridge-rope", use_amw=False)
NOVA_1B_NOPE = replace(NOVA_1B, name="nova1b-m3-amw-nope", anchor_position="nope")

CANDIDATES = {
    NOVA_1B.name: NOVA_1B,
    NOVA_1B_NO_AMW.name: NOVA_1B_NO_AMW,
    NOVA_1B_NOPE.name: NOVA_1B_NOPE,
}


def main() -> None:
    for cfg in CANDIDATES.values():
        print(cfg.name, cfg.parameter_breakdown())
    expected = 999_873_280
    actual = NOVA_1B.parameter_breakdown()["total"]
    if actual != expected:
        raise SystemExit(f"Nova 1B analytical contract drift: {actual:,} != {expected:,}")
    if NOVA_1B.schedule() != ("mamba3", "mamba3", "mamba3", "attention") * 8:
        raise SystemExit("Nova schedule drift")
    print(f"Nova 1B analytical contract OK: {actual:,} parameters")


if __name__ == "__main__":
    main()
