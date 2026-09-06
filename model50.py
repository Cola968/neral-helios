"""Own 50M model: pad attention head 52 -> 56 only inside parameter-free SDPA."""
import math
import torch.nn.functional as F
from model import Neral20M, CausalSelfAttention
from architectures import FIFTY


class PaddedAttention(CausalSelfAttention):
    """Tensor-core-friendly SDPA padding without changing model weights or RoPE geometry."""

    def _sdpa(self, q, k, v, is_causal: bool):
        padding = (-self.head_dim) % 8
        q_pad, k_pad, v_pad = (F.pad(a, (0, padding)) for a in (q, k, v))
        y = F.scaled_dot_product_attention(
            q_pad,
            k_pad,
            v_pad,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=1 / math.sqrt(self.head_dim),
        )
        return y[..., : self.head_dim]


class Neral50M(Neral20M):
    def __init__(self, cfg=None):
        super().__init__(cfg or FIFTY)
        for block in self.blocks:
            attention = PaddedAttention(self.cfg)
            attention.load_state_dict(block.attn.state_dict())
            block.attn = attention
