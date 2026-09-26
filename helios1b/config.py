from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class HeliosConfig:
    """Architecture contract for the new Neral Helios generation.

    The default is the production-scale ~1B dense decoder. Use tiny_proxy()
    for architecture and training-stack validation without allocating 1B weights.
    """

    vocab_size: int = 32_768
    d_model: int = 1_536
    n_layers: int = 32
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 128
    ffn_hidden: int = 5_120
    max_seq_len: int = 4_096
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-5
    dropout: float = 0.0
    qk_norm: bool = True
    z_loss_coeff: float = 1e-4
    nope_every: int = 0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if self.vocab_size <= 0 or self.n_layers <= 0 or self.ffn_hidden <= 0:
            raise ValueError("vocab_size, n_layers and ffn_hidden must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.nope_every < 0:
            raise ValueError("nope_every must be >= 0")

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def groups_per_kv_head(self) -> int:
        return self.n_heads // self.n_kv_heads

    def layer_uses_rope(self, layer_index: int) -> bool:
        if not 0 <= layer_index < self.n_layers:
            raise IndexError(layer_index)
        return self.nope_every <= 0 or (layer_index + 1) % self.nope_every != 0

    def parameter_count(self) -> int:
        embeddings = self.vocab_size * self.d_model

        attention = (
            self.d_model * (self.n_heads * self.head_dim)
            + self.d_model * self.kv_dim
            + self.d_model * self.kv_dim
            + self.d_model * self.d_model
        )

        mlp = 3 * self.d_model * self.ffn_hidden

        norms = 2 * self.d_model
        if self.qk_norm:
            norms += 2 * self.head_dim

        per_layer = attention + mlp + norms
        final_norm = self.d_model

        total = embeddings + self.n_layers * per_layer + final_norm
        if not self.tie_embeddings:
            total += self.vocab_size * self.d_model
        return total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def tiny_proxy(cls) -> "HeliosConfig":
        return cls(
            vocab_size=512,
            d_model=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=32,
            ffn_hidden=352,
            max_seq_len=128,
            z_loss_coeff=1e-4,
        )

    def with_nope_ablation(self, every: int = 4) -> "HeliosConfig":
        if every <= 0:
            raise ValueError("every must be positive")
        return replace(self, nope_every=every)
