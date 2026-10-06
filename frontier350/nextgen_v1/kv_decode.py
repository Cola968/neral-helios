"""Research-only incremental inference using the existing Neral 350M weights.

No architecture or checkpoint changes. Activate in the preview only after real
checkpoint logit/greedy parity passes; inference only, not training throughput.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from model350 import rope


def cached_last_logits(model, tokens: torch.Tensor, past=None):
    """Return last-position logits and one immutable-per-call K/V cache per layer.

    First call: entire prompt, causal attention. Following calls: ONLY new tokens;
    attention must be non-causal when query length is 1 and keys include the past.
    Never use this path while the model is in training mode.
    """
    if model.training:
        raise ValueError("KV decoder requires eval mode")
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
        raise ValueError("expected batch-one nonempty token IDs")
    if past is not None and (len(past) != len(model.blocks) or tokens.shape[1] != 1):
        raise ValueError("cached decode requires one new token and complete layer cache")
    offset = 0 if past is None else past[0][0].size(-2)
    if offset + tokens.shape[1] > model.cfg.max_seq_len:
        raise ValueError("context ceiling exceeded")
    x = model.tok_emb(tokens)
    next_cache = []
    for i, block in enumerate(model.blocks):
        h = block.n1(x)
        batch, length, _ = h.shape
        attn = block.attn
        q = attn.q(h).view(batch, length, model.cfg.n_heads, model.cfg.head_dim).transpose(1, 2)
        k = attn.k(h).view(batch, length, model.cfg.n_kv_heads, model.cfg.head_dim).transpose(1, 2)
        v = attn.v(h).view(batch, length, model.cfg.n_kv_heads, model.cfg.head_dim).transpose(1, 2)
        q, k = attn.qn(q), attn.kn(k)
        q, k = rope(q, k, model.cfg.rope_theta, model.cfg.max_seq_len, offset=offset)
        if past is not None:
            old_k, old_v = past[i]
            if old_k.size(-2) != offset or old_v.size(-2) != offset:
                raise ValueError("inconsistent layer cache lengths")
            k = torch.cat((old_k, k), dim=-2)
            v = torch.cat((old_v, v), dim=-2)
        next_cache.append((k, v))
        try:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None,
                                                dropout_p=0.0, enable_gqa=True)
        except (TypeError, RuntimeError):
            # Same compatibility fallback as model350.GQA.forward.
            kr = k.repeat_interleave(model.cfg.gqa_groups, dim=1)
            vr = v.repeat_interleave(model.cfg.gqa_groups, dim=1)
            y = F.scaled_dot_product_attention(q, kr, vr, is_causal=past is None,
                                                dropout_p=0.0)
        x = x + attn.o(y.transpose(1, 2).contiguous().view(batch, length, model.cfg.d_model))
        x = x + block.ffn(block.n2(x))
    return model.lm_head(model.norm(x))[:, -1, :], tuple(next_cache)
