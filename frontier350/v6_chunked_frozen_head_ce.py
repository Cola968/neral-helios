from __future__ import annotations

"""Experimental low-VRAM CE for a frozen LM head.

NOT wired into training. V6 may use this only after the equivalence/endurance gates in
V6_STREAMGUARD_DESIGN.md pass. The head is frozen in Neral continuation training, so the
custom backward only needs a gradient for hidden states.

The implementation avoids materializing the full [tokens, vocab] logits matrix. Forward
computes logsumexp over vocabulary chunks. Backward recomputes the same chunks and sums
softmax(logits) @ W - W[target]. Floating-point execution order differs from standard
PyTorch CE, so runner-side BF16 equivalence testing is mandatory before activation.
"""

import torch
import torch.nn.functional as F


class _FrozenHeadChunkedCESum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                chunk_size: int, ignore_index: int):
        if weight.requires_grad:
            raise ValueError("chunked frozen-head CE requires a frozen head")
        if hidden.shape[:-1] != targets.shape:
            raise ValueError("targets must match hidden token dimensions")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        h = hidden.reshape(-1, hidden.shape[-1])
        t = targets.reshape(-1).long()
        vocab = int(weight.shape[0])
        if weight.ndim != 2 or weight.shape[1] != h.shape[1]:
            raise ValueError("head weight shape mismatch")
        valid = t.ne(ignore_index)
        if bool(((t[valid] < 0) | (t[valid] >= vocab)).any()):
            raise ValueError("target outside vocabulary")

        # Online logsumexp in FP32; the linear itself keeps the incoming hidden/weight
        # dtype, matching the intended BF16 head matmul on CUDA.
        running_max = torch.full((h.shape[0],), -torch.inf, device=h.device, dtype=torch.float32)
        running_sum = torch.zeros_like(running_max)
        target_logit = torch.zeros_like(running_max)

        for start in range(0, vocab, int(chunk_size)):
            end = min(vocab, start + int(chunk_size))
            logits = F.linear(h, weight[start:end]).float()
            chunk_max = logits.max(dim=1).values
            new_max = torch.maximum(running_max, chunk_max)
            running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
                logits - new_max[:, None]
            ).sum(dim=1)
            running_max = new_max
            hit = valid & t.ge(start) & t.lt(end)
            if bool(hit.any()):
                target_logit[hit] = logits[hit, t[hit] - start]

        lse = running_max + torch.log(running_sum)
        per_token = torch.where(valid, lse - target_logit, torch.zeros_like(lse))
        ctx.save_for_backward(hidden, weight, targets, lse)
        ctx.chunk_size = int(chunk_size)
        ctx.ignore_index = int(ignore_index)
        return per_token.sum()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, targets, lse = ctx.saved_tensors
        h = hidden.reshape(-1, hidden.shape[-1])
        t = targets.reshape(-1).long()
        valid = t.ne(ctx.ignore_index)
        vocab = int(weight.shape[0])

        # Accumulate in FP32. This is the exact analytical CE gradient for the logits
        # produced in forward; final casting follows the hidden input dtype.
        grad_h = torch.zeros((h.shape[0], h.shape[1]), device=h.device, dtype=torch.float32)
        scale = grad_output.float()
        for start in range(0, vocab, ctx.chunk_size):
            end = min(vocab, start + ctx.chunk_size)
            logits = F.linear(h, weight[start:end]).float()
            probs = torch.exp(logits - lse[:, None])
            probs[~valid] = 0.0
            grad_h.add_(probs @ weight[start:end].float())

        if bool(valid.any()):
            grad_h[valid] -= weight[t[valid]].float()
        grad_h.mul_(scale)
        return grad_h.reshape_as(hidden).to(hidden.dtype), None, None, None, None


def frozen_head_chunked_ce_sum(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                               *, chunk_size: int = 1024, ignore_index: int = -100) -> torch.Tensor:
    return _FrozenHeadChunkedCESum.apply(hidden, weight, targets, int(chunk_size), int(ignore_index))


def reference_equivalence(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                          *, chunk_size: int = 1024, ignore_index: int = -100) -> dict[str, float]:
    """One-batch diagnostic; caller should repeat across >=100 fixed runner batches."""
    if weight.requires_grad:
        raise ValueError("weight must be frozen")
    h_ref = hidden.detach().clone().requires_grad_(True)
    h_new = hidden.detach().clone().requires_grad_(True)

    ref_logits = F.linear(h_ref, weight)
    ref = F.cross_entropy(ref_logits.reshape(-1, ref_logits.shape[-1]), targets.reshape(-1),
                          ignore_index=ignore_index, reduction="sum")
    new = frozen_head_chunked_ce_sum(h_new, weight, targets, chunk_size=chunk_size,
                                     ignore_index=ignore_index)
    ref.backward(); new.backward()
    g0 = h_ref.grad.float().reshape(-1)
    g1 = h_new.grad.float().reshape(-1)
    cosine = F.cosine_similarity(g0, g1, dim=0).item()
    rel_norm = abs(g0.norm().item() - g1.norm().item()) / max(g0.norm().item(), 1e-12)
    return {
        "loss_abs_diff": abs(float(ref.detach()) - float(new.detach())),
        "grad_cosine": float(cosine),
        "grad_rel_norm_diff": float(rel_norm),
    }
