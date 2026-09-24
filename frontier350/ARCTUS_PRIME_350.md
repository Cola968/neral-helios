# Arctus Prime 350M — research candidate

Status: **research only / not promoted**.

The existing `frontier/neral350-v1` model remains the production control and its current checkpoints are untouched. Prime is deliberately shape-incompatible and must be trained from scratch for any fair quality comparison.

## Why this exists

Frontier 350 v1 is the measured RTX-4070 hardware winner and already contains several strong choices: 16K Neral BPE, tied embeddings, Pre-RMSNorm, Q/K RMSNorm, RoPE, SwiGLU, bias-free linear layers, zero dropout, residual-output scaling and GQA.

Prime does **not** pretend those are new. It changes only the parameter allocation and execution layout where there is a credible quality/efficiency hypothesis.

## Locked candidate

| Property | Frontier 350 v1 | Arctus Prime 350 |
|---|---:|---:|
| Parameters | 349,825,152 | **348,975,616** |
| Layers | 24 | **28** |
| Width | 1152 | **1024** |
| Q heads | 18 | **16** |
| KV heads | 2 | **4** |
| Head dim | 64 | **64** |
| GQA ratio | 9:1 | **4:1** |
| SwiGLU hidden | 3136 | **3008** |
| Vocabulary | 16,384 | **16,384** |
| Context contract | 8192 | **8192** |
| RoPE theta | 250,000 | **250,000** |
| Q/K RMSNorm | yes | **yes** |
| tied embedding/head | yes | **yes** |
| residual scaling | yes | **yes** |
| linear bias | no | **no** |
| QKV execution | separate in reference; fused fast path exists | **fused from start** |
| SwiGLU gate/up | separate in reference; fused fast path exists | **fused from start** |

## Quality hypothesis

Prime spends essentially the same parameter budget differently:

1. **28 instead of 24 transformer stages** gives four additional attention/non-linear transformations.
2. **4:1 instead of 9:1 GQA** gives substantially more K/V representation capacity for context retrieval and long dependencies.
3. **1024 width and 64-d heads** stay clean for Tensor-Core/SDPA kernels.
4. **3008 SwiGLU** preserves a strong MLP budget while paying for the extra depth and K/V capacity.
5. The fused QKV and gate/up layout is the native execution form rather than a post-hoc conversion.

These are hypotheses, not measured quality claims.

## Things intentionally NOT added

Prime does not default to MoE, MLA, Differential Attention, sliding-window attention, learned residual gates, FP8, exotic positional encodings or other research features. At ~350M on one RTX 4070 those can add kernel complexity or alter quality without sufficient evidence. They should be isolated ablations, never silently stacked into the base candidate.

Full causal attention remains the quality-control default. A local/global hybrid may be benchmarked later for long-context throughput, but only as a separate variant.

## Promotion protocol

Prime may replace v1 only after a matched bake-off.

### Gate A — architecture integrity
- exact analytical parameter count: **348,975,616**
- same locked 16K Neral tokenizer
- no foreign weights or checkpoint transplant
- finite forward/backward
- tied embedding/head verified
- deterministic checkpoint/resume test

### Gate B — RTX 4070 execution
Measure v1 and Prime with the same sequence length, tokens/update, dtype, compiler mode and warmup policy.

Record:
- end-to-end tokens/s
- peak allocated/reserved VRAM
- compile time and graph breaks
- train loss finite / gradient finite

Do not call Prime faster or slower before this measurement exists.

### Gate C — matched learning pilot
Train from scratch on the same shuffled token stream and seed. Minimum useful checkpoints: 25M and 100M tokens.

Compare:
- validation CE / BPB
- loss slope per token
- EOS/repetition health
- fixed German/English probes
- math/code fixed probes
- NIS2 only when generations are mature enough for it to be meaningful

### Promotion rule
Prime should not win merely because of theory. A reasonable promotion requires one of:

- **>=3% relative validation/BPB improvement** with <=10% throughput loss, or
- **>=5% quality-score improvement** with <=15% throughput loss, or
- equal quality with a material speed/VRAM advantage.

If none of those are met, keep Frontier 350 v1.

## Expected—not measured—direction

The design is intended to trade a small amount of raw throughput for higher quality per parameter, especially context/retrieval and multi-stage reasoning. No percentage gain is recorded as fact until the matched pilot is complete.
