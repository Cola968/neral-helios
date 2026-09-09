# Arctus 350M architecture search

The safest way to build a better 350M model is not to declare one speculative architecture the winner. This search keeps the current Frontier v1 as the control and tests two parameter-matched challengers.

## Candidates

| ID | Layers | d_model | Q/KV | head_dim | SwiGLU | Params | Purpose |
|---|---:|---:|---:|---:|---:|---:|---|
| V1 control | 24 | 1152 | 18/2 | 64 | 3136 | 349,825,152 | measured RTX-4070 hardware winner |
| Prime-B | 24 | 1152 | 18/3 | 64 | 3072 | 348,055,680 | low-risk quality/speed balance |
| Prime-Q | 28 | 1024 | 16/4 | 64 | 3008 | 348,975,616 | quality-first depth + stronger K/V capacity |

All use the same 16,384-token Neral BPE, 8K context contract, RoPE, Pre-RMSNorm, Q/K RMSNorm, SwiGLU, tied embeddings, bias-free linear layers and scaled residual initialization.

## Why two challengers

### Prime-B
Keeps the exact v1 depth/width geometry that already won on the RTX 4070, but moves from 9:1 to 6:1 GQA by adding a third KV head. The FFN is reduced from 3136 to 3072 to stay near the same parameter budget and retain a hardware-friendly hidden size. This is the candidate most likely to preserve v1 throughput.

### Prime-Q
Moves to 28x1024 and 4:1 GQA. It gives up some width and adds four transformer stages plus much more K/V capacity. This is the stronger quality hypothesis, but it may be slower because of the extra serial layers.

## Successive-halving bake-off

Do not spend 100M tokens on every design immediately.

1. **Synthetic execution gate** — identical seq=512, BF16, compile mode, tokens/update. Reject OOM, numerical failures or severe throughput regressions.
2. **5M-token learning smoke** — same seed, exact token stream and LR schedule. Reject unstable or clearly worse loss slope.
3. **25M-token matched pilot** — compare validation CE/BPB, throughput, peak VRAM and generation health.
4. **100M-token finalist** — only the best challenger versus V1 control gets the expensive quality gate.

## Decision metric

No single scalar should decide the architecture. Record a Pareto tuple:

`(validation_BPB, quality_score, tok_per_s, peak_VRAM_GB, compile_stability)`

A challenger is promotable only if it dominates v1 on quality with an acceptable hardware cost, or matches quality while materially improving hardware efficiency.

Suggested hard guardrails:

- >15% throughput loss: reject unless quality gain is exceptional.
- >10% peak-VRAM increase: require a clear quality gain.
- non-finite loss/gradients, broken resume, or compile instability: reject.
- <=1% quality movement: treat as noise until replicated.

## Not part of the base search

MoE, MLA, Differential Attention, local/global hybrid attention, FP8, speculative auxiliary heads and alternative positional encodings remain separate ablations. Stacking them before the base geometry is settled would make causal attribution impossible.

The current Frontier v1 checkpoint is never modified by this search.
