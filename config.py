from dataclasses import dataclass, asdict


@dataclass
class NeralConfig:
    vocab_size: int = 260
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ffn_hidden: int = 1536
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    dropout: float = 0.0
    rms_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    def to_dict(self):
        return asdict(self)
