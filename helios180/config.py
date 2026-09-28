from __future__ import annotations
from dataclasses import asdict, dataclass, replace

@dataclass(frozen=True)
class Helios180Config:
    """Fast local chatbot architecture for Neral Chat v1."""
    vocab_size: int = 32_768
    d_model: int = 768
    n_layers: int = 24
    n_heads: int = 6
    n_kv_heads: int = 3
    head_dim: int = 128
    ffn_hidden: int = 2_048
    max_seq_len: int = 4_096
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-5
    dropout: float = 0.0
    qk_norm: bool = True
    z_loss_coeff: float = 1e-4
    tie_embeddings: bool = True
    native_gqa: bool = True

    def __post_init__(self):
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def groups_per_kv_head(self) -> int:
        return self.n_heads // self.n_kv_heads

    def parameter_count(self) -> int:
        emb = self.vocab_size * self.d_model
        attn = (
            self.d_model * (self.n_heads * self.head_dim)
            + 2 * self.d_model * self.kv_dim
            + self.d_model * self.d_model
        )
        mlp = 3 * self.d_model * self.ffn_hidden
        norms = 2 * self.d_model + (2 * self.head_dim if self.qk_norm else 0)
        total = emb + self.n_layers * (attn + mlp + norms) + self.d_model
        if not self.tie_embeddings:
            total += emb
        return total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def tiny_proxy(cls) -> "Helios180Config":
        return cls(vocab_size=512, d_model=128, n_layers=2, n_heads=4,
                   n_kv_heads=2, head_dim=32, ffn_hidden=352, max_seq_len=128)

    def with_native_gqa(self, enabled: bool) -> "Helios180Config":
        return replace(self, native_gqa=enabled)
