"""Opt-in execution experiments. Canonical parameters, norms and optimizer stay intact."""
import hashlib
import torch
import torch.nn.functional as F
from model350 import GQA, SwiGLU, rope


class AuditForward:
    """Diagnostic-only sampled logits keyed by input row, independent of packing.

    The CPU copies intentionally synchronize; never use an audit as a speed test.
    Full trainable gradients are audited separately by the trainer.
    """
    def __init__(self, run):
        self.run = run
        self.rows = {}

    def __call__(self, x, y=None):
        logits, loss = self.run(x, y)
        samples = logits.detach()[:, ::max(1,x.size(1)//3),
            ::max(1,logits.size(-1)//128)].float().cpu()
        for row, sample in zip(x.detach().cpu(), samples):
            key = hashlib.sha256(row.contiguous().numpy().tobytes()).hexdigest()
            self.rows[key] = sample
        return logits, loss


class FusedCanonicalGQA(GQA):
    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = F.linear(x, torch.cat((self.q.weight, self.k.weight, self.v.weight))).split(
            (self.cfg.d_model, self.cfg.kv_dim, self.cfg.kv_dim), dim=-1)
        q = q.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q, k = self.qn(q), self.kn(k)
        q, k = rope(q, k, self.cfg.rope_theta, self.cfg.max_seq_len)
        try:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0, enable_gqa=True)
        except (TypeError, RuntimeError):
            k = k.repeat_interleave(self.cfg.gqa_groups, 1)
            v = v.repeat_interleave(self.cfg.gqa_groups, 1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model))


class FusedCanonicalSwiGLU(SwiGLU):
    def forward(self, x):
        gate, up = F.linear(x, torch.cat((self.gate.weight, self.up.weight))).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def enable_fused_execution(model):
    """Change execution only; retain actual Parameter objects and state_dict keys.

    Concatenations are differentiable and deliberately not cached: caching trainable
    weights would silently use stale parameters after AdamW steps.
    """
    if model.cfg.dropout != 0:
        raise ValueError('equivalence experiment requires the locked zero-dropout model')
    for block in model.blocks:
        block.attn.__class__ = FusedCanonicalGQA
        block.ffn.__class__ = FusedCanonicalSwiGLU


def transfer_update(batches, device):
    """One pinned H2D transfer; views own the slab until backward has completed."""
    sizes = [x.size(0) for x, _ in batches]
    slab = torch.stack((torch.cat([x for x, _ in batches]),
                        torch.cat([y for _, y in batches])))
    if torch.device(device).type == 'cuda':
        slab = slab.pin_memory().to(device, non_blocking=True)
    else:
        slab = slab.to(device)
    return list(zip(slab[0].split(sizes), slab[1].split(sizes)))


def execution_groups(batches, counts, batch, replay_boost, pack, fuse_replay):
    """Keep sampling order fixed; >Pack9 combines main and replay within ONE update.

    batches contains three replay batches followed by the production Pack9 main
    batch (when present). Valid-token counts, not physical padding, set weights.
    """
    if pack not in (9, 10, 11, 12) or len(counts) < 3:
        raise ValueError('requires Pack9-12 and all three replay references')
    if pack > 9 and not fuse_replay:
        raise ValueError('mixed-domain packs require replay fusion')
    if pack == 9:
        groups = [list(range(3))] if fuse_replay else [[0], [1], [2]]
        if len(batches) > 3:
            groups.append([3])
        for indices in groups:
            count = sum(counts[i] if i < 3 else sum(counts[3:]) for i in indices)
            weight = replay_boost if indices[0] < 3 else 1.0
            yield torch.cat([batches[i][0] for i in indices]), torch.cat([batches[i][1] for i in indices]), count * weight, None
        return
    # Regroup the same sampled rows. No sampling across optimizer-update boundaries.
    xs = [x for x, _ in batches[3:]] + [x for x, _ in batches[:3]]
    ys = [y for _, y in batches[3:]] + [y for _, y in batches[:3]]
    weights = [torch.full_like(y, 1.0, dtype=torch.float32) for _, y in batches[3:]]
    weights += [torch.full_like(y, replay_boost, dtype=torch.float32) for _, y in batches[:3]]
    x, y, w = torch.cat(xs), torch.cat(ys), torch.cat(weights)
    for start in range(0, x.size(0), pack * batch):
        end = start + pack * batch
        yield x[start:end], y[start:end], None, w[start:end]


def backward_update(run, groups, denominator, device):
    loss_acc = torch.zeros((), device=device, dtype=torch.float32)
    finite_flags = []
    for x, y, mass, weights in groups:
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                            enabled=torch.device(device).type == 'cuda'):
            if weights is None:
                _, loss = run(x, y)
                objective = loss * (mass / denominator)
            else:
                logits, _ = run(x)
                per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1), reduction='none').view_as(y)
                objective = (per_token * weights).sum() / denominator
        finite_flags.append(torch.isfinite(objective))
        objective.backward()
        loss_acc.add_(objective.detach().float())
    return loss_acc, finite_flags
