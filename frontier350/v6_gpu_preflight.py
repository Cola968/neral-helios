from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F

from model350 import FRONTIER_350
from streamguard_v6_policy import choose_profile
from v6_chunked_frozen_head_ce import frozen_head_chunked_ce_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required')
    gpu = torch.cuda.get_device_name(0)
    if 'RTX 4070' not in gpu:
        raise RuntimeError(f'expected RTX 4070, got {gpu}')

    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    profile = choose_profile(free_mib=free // 1048576, reserved_mib=reserved // 1048576)

    # The chunked implementation intentionally accumulates logsumexp/loss in FP32.
    # Therefore the fidelity gate must compare against standard CE on the exact same
    # BF16 matmul logits promoted to FP32 before CE. Comparing the returned BF16 CE
    # scalar directly introduces reduction/quantization error that is unrelated to
    # the chunking algorithm. We still record that native-BF16 difference as a
    # diagnostic, but it is not the production fidelity gate.
    worst_loss = 0.0
    worst_native_bf16_loss = 0.0
    min_cos = 1.0
    worst_norm = 0.0
    batches = 100

    for seed in range(batches):
        torch.manual_seed(70000 + seed)
        torch.cuda.manual_seed_all(70000 + seed)
        n, d, v = 128, FRONTIER_350.d_model, FRONTIER_350.vocab_size
        w = (torch.randn(v, d, device='cuda', dtype=torch.bfloat16) * 0.02).requires_grad_(False)
        t = torch.randint(0, v, (n,), device='cuda')
        t[::17] = -100
        h = torch.randn(n, d, device='cuda', dtype=torch.bfloat16)
        h0 = h.clone().requires_grad_(True)
        h1 = h.clone().requires_grad_(True)

        logits_ref = F.linear(h0, w)
        ref = F.cross_entropy(logits_ref.float(), t, ignore_index=-100, reduction='sum')
        native_bf16_ref = F.cross_entropy(logits_ref.detach(), t, ignore_index=-100, reduction='sum')
        q = frozen_head_chunked_ce_sum(h1, w, t, chunk_size=1024, ignore_index=-100)

        ref.backward()
        q.backward()
        torch.cuda.synchronize()

        valid = max(1, int(t.ne(-100).sum()))
        g0 = h0.grad.float().flatten()
        g1 = h1.grad.float().flatten()
        worst_loss = max(worst_loss, abs(float(ref.detach()) - float(q.detach())) / valid)
        worst_native_bf16_loss = max(
            worst_native_bf16_loss,
            abs(float(native_bf16_ref.detach()) - float(q.detach())) / valid,
        )
        min_cos = min(min_cos, float(F.cosine_similarity(g0, g1, dim=0)))
        worst_norm = max(
            worst_norm,
            abs(float(g0.norm()) - float(g1.norm())) / max(float(g0.norm()), 1e-12),
        )

    passed = worst_loss <= 2e-4 and min_cos >= 0.99999 and worst_norm <= 5e-4
    report = {
        'schema': 2,
        'gpu': gpu,
        'torch': torch.__version__,
        'profile': profile.name,
        'profile_chunks': list(profile.chunks),
        'free_gib_start': free / 1073741824,
        'fidelity': {
            'batches': batches,
            'reference': 'standard CE on identical BF16 linear logits promoted to FP32',
            'worst_mean_loss_abs_diff': worst_loss,
            'native_bf16_ce_worst_mean_loss_abs_diff_info': worst_native_bf16_loss,
            'min_grad_cosine': min_cos,
            'worst_grad_rel_norm_diff': worst_norm,
            'pass': passed,
        },
        'note': 'isolated head-level GPU preflight; no checkpoint is loaded or modified',
    }

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit('V6 BF16 fidelity gate failed')


if __name__ == '__main__':
    main()
