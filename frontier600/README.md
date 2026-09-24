# Neral 600M Architecture Search

Status: **research only — no winner has been promoted**.

This branch searches for a dense ~600M successor to Arctus 350M, trained completely from scratch with Neral's own 16,384-token Byte-BPE tokenizer. No foreign model weights, base-model checkpoints, vocabulary, or merges are used.

The search is explicitly optimized for one **NVIDIA RTX 4070 12 GB**, but model quality has priority over raw tokens/s.

## Baseline

Arctus Frontier 350 v1 is the control:

- 349,825,152 trainable parameters
- 24 layers, d_model 1152
- 18 Q / 2 KV heads, head_dim 64 (9:1 GQA)
- SwiGLU 3136
- Pre-RMSNorm + Q/K RMSNorm
- RoPE theta 250,000
- tied 16K Neral Byte-BPE embedding/LM head
- 8K context contract
- BF16, SDPA/Flash-compatible attention, torch.compile/TorchInductor
- short measured RTX-4070 speed record elsewhere in the project: ~19.2k BPE tok/s. This is a short benchmark, not a sustained long-run claim.

## Phase A — ten dense candidates

Exact counts include the tied token embedding once, every Q/K/V/O projection, every SwiGLU gate/up/down matrix, block RMSNorms, Q/K RMSNorm vectors, and final RMSNorm.

| ID | Layers | d_model | Q/KV | GQA | head | SwiGLU | Embedding | Attention/layer | FFN/layer | Norm/layer | Exact trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W24 | 24 | 1536 | 24/6 | 4:1 | 64 | 3904 | 25,165,824 | 5,898,240 | 17,989,632 | 3,200 | **598,553,088** |
| W26 | 26 | 1536 | 24/6 | 4:1 | 64 | 3520 | 25,165,824 | 5,898,240 | 16,220,160 | 3,200 | **600,328,960** |
| W28 | 28 | 1536 | 24/6 | 4:1 | 64 | 3200 | 25,165,824 | 5,898,240 | 14,745,600 | 3,200 | **603,284,480** |
| B28 | 28 | 1344 | 21/3 | 7:1 | 64 | 4096 | 22,020,096 | 4,128,768 | 16,515,072 | 2,816 | **600,127,808** |
| B30 | 30 | 1280 | 20/4 | 5:1 | 64 | 4032 | 20,971,520 | 3,932,160 | 15,482,880 | 2,688 | **603,504,640** |
| B32 | 32 | 1280 | 20/5 | 4:1 | 64 | 3648 | 20,971,520 | 4,096,000 | 14,008,320 | 2,688 | **600,397,056** |
| D36 | 36 | 1152 | 18/3 | 6:1 | 64 | 3776 | 18,874,368 | 3,096,576 | 13,049,856 | 2,432 | **600,234,624** |
| D40 | 40 | 1024 | 16/4 | 4:1 | 64 | 3904 | 16,777,216 | 2,621,440 | 11,993,088 | 2,176 | **601,446,400** |
| F27 | 27 | 1152 | 18/2 | 9:1 | 64 | 5376 | 18,874,368 | 2,949,120 | 18,579,456 | 2,432 | **600,212,736** |
| Q36 | 36 | 1280 | 20/5 | 4:1 | 64 | 3136 | 20,971,520 | 4,096,000 | 12,042,240 | 2,688 | **602,046,208** |

The final RMSNorm adds d_model parameters to each total and is included in the exact total above.

## Why these geometries

The set deliberately spans four hypotheses instead of making tiny cosmetic changes:

1. **Wide/shallow (W24/W26/W28)** — fewer serial transformer blocks and larger GEMMs. Strong RTX-4070 throughput hypothesis; still 4:1 GQA.
2. **Balanced (B28/B30/B32)** — progressively more depth while retaining substantial width and FFN capacity. B32 is the pre-hardware sweet-spot hypothesis, not a winner.
3. **Deep/narrow (D36/D40)** — Prime-like emphasis on transformation depth and reasoning at the cost of more kernel launches/serial dependency.
4. **Allocation controls (F27/Q36)** — F27 spends unusually heavily on the MLP with aggressive GQA; Q36 spends on depth + 4:1 GQA and is the pre-hardware max-quality hypothesis.

All candidates keep head_dim=64. This is both a model-design choice and an RTX-4070 execution choice: it maps cleanly to SDPA/Flash kernels and avoids padding tricks.

## Locked common architecture decisions

For the base geometry search, every candidate uses:

- dense decoder-only causal Transformer
- 16,384-token Neral Byte-BPE
- Pre-RMSNorm
- Q/K RMSNorm
- SwiGLU
- GQA
- RoPE
- tied input/output embedding matrix
- no linear bias
- zero pretraining dropout
- scaled residual-output initialization (`0.02 / sqrt(2L)` for attention O and MLP down projections)
- fused QKV execution layout
- fused SwiGLU gate/up execution layout
- PyTorch SDPA / Flash-compatible attention
- BF16
- torch.compile / TorchInductor

### RoPE

The search candidate default is theta **500,000** to leave more positional headroom for a later 8K -> 16K extension. This is **not yet a measured improvement over Arctus theta=250,000**. Once geometry is selected, 250K vs 500K must be a controlled positional ablation before the final long run.

### Full attention first

Full causal attention is the quality-control baseline. Sliding/local attention, hybrid local/global masks and FlexAttention are not stacked into Phase C/D because that would confound geometry attribution. After a geometry winner exists, a separate 3-local:1-global or similar ablation can be tested for 8K/16K efficiency.

### Features intentionally not defaulted

- **DeepNorm:** not default. Pre-RMSNorm + residual-output scaling is simpler and already stable in the project at these depths.
- **MuP:** potentially valuable for hyperparameter transfer/scaling studies, but it changes parameterization/training semantics. Treat as a later training-recipe ablation, not as an unmeasured geometry feature.
- **variable FFN by layer:** rejected for the initial search; it complicates compilation and causal attribution without strong project-specific evidence.
- **embedding scaling:** not added without a measured need.
- **FP8:** not a baseline. BF16 is the correctness/control path; any Ada FP8 path needs a strict loss/gradient equivalence and speed gate.
- **8-bit optimizer:** not baseline because quality/stability outrank VRAM. It becomes a fallback only if BF16 + fused AdamW + checkpointing cannot fit a finalist.
- **optimizer offload:** last resort because PCIe traffic can destroy throughput.
- **MoE:** not a base candidate. A single 12 GB GPU still has to store all expert parameters/optimizer state, and routing/fragmentation adds complexity. A sparse experiment may follow only if dense 600M cannot meet the desired Pareto point.

## Phase B — exact parameters + analytical FLOPs

`architecture_search600.py` computes exact parameter accounting and an approximate forward/training FLOP model for every candidate. Multiply-add is counted as two FLOPs. The estimate includes Q/K/V/O and FFN GEMMs, causal attention score/value work, and the LM-head projection. Norm/RoPE/activation elementwise FLOPs are omitted as second-order terms.

The FLOP model is analytical — **not measured GPU throughput**.

## Phase C — real RTX-4070 execution gate

Pre-hardware finalists:

- W26 — wide/shallow hardware candidate
- B32 — balanced sweet-spot hypothesis
- D36 — Prime-like deep candidate
- Q36 — max-quality/depth candidate

`bench600.py` performs a real BF16 optimizer-inclusive training-path benchmark at identical sequence length and **16,384 effective tokens/update**, trying microbatch 1 and 2. It records measured tokens/s, peak allocated/reserved VRAM, finite loss/gradient state and compile configuration. If a no-checkpoint profile OOMs it retries with selective block checkpointing.

Do not describe any 600M tokens/s or VRAM value as measured until this benchmark has actually completed on the self-hosted RTX 4070.

## Phase D — matched learning-curve pilot

`train_pilot600.py` trains finalists from scratch on the exact same Neral BPE token shards using:

- same seed
- same sampled token stream
- same sequence length
- same effective tokens/update
- same optimizer settings
- same warmup policy
- same fixed validation samples

The initial gate is 5M tokens per finalist. It records train loss, validation CE/perplexity, gradient norm, measured end-to-end tok/s, peak VRAM and stability. BPB is recorded only when a corpus bytes/token value is explicitly supplied; it is then labeled as an estimate.

A 5M pilot is useful for rejecting unstable or obviously weak geometries, not for claiming mature reasoning quality. A close race must continue to 25M and preferably 100M before promotion.

## Phase E — decision rule

Required weighted score:

- 40% model quality / learning curve
- 20% quality per parameter
- 15% training speed
- 10% VRAM
- 10% long-context efficiency
- 5% inference efficiency

The scalar score is a decision aid, not permission to ignore Pareto failures. In particular, a faster architecture must not win if validation quality is materially worse.

Long-context efficiency is evaluated separately after the geometry gate using 8K analytical attention/KV-cache cost plus an actual long-sequence execution test for the surviving two candidates. It must not be fabricated from the 512-token speed run.

## Pre-hardware hypotheses — NOT winners

### A: RTX-4070 Sweet Spot hypothesis — B32

- **600,397,056** trainable parameters
- vocab 16,384
- d_model 1280
- 32 layers
- 20 Q heads / 5 KV heads
- head_dim 64
- 4:1 GQA
- SwiGLU hidden 3648
- Pre-RMSNorm + Q/K RMSNorm
- full causal GQA/SDPA
- RoPE, theta 500K candidate
- 8K initial context contract
- tied embedding/head
- residual-output scaling
- BF16
- recommended starting profile: microbatch 1, grad accumulation 32 at seq512 = 16,384 tokens/update; test MB2/GA16
- selective checkpoint every second block as safe baseline, remove if hardware measurements show it is unnecessary

Why it is attractive: 1280 and all relevant head dimensions are clean GEMM sizes, 32 layers provide more compositional depth than Arctus, and 4:1 GQA materially increases K/V capacity compared with Arctus 9:1 without reverting to expensive MHA.

### B: MAX QUALITY hypothesis — Q36

- **602,046,208** trainable parameters
- d_model 1280
- 36 layers
- 20 Q / 5 KV, head_dim64, 4:1 GQA
- SwiGLU hidden 3136
- otherwise same base architecture

Q36 buys four extra transformer stages by reducing FFN width. It is expected to be slower and may improve multi-stage reasoning, but this remains a hypothesis until matched learning curves exist.

## Estimated hardware ranges before Phase C

These are **engineering estimates, not measurements**. They are intentionally broad because compile quality, optimizer-state dtype, checkpoint pattern and microbatch determine the real result.

| Candidate | starting MB | effective tokens/update | estimated peak VRAM | estimated BPE tok/s | estimated speed vs measured Arctus short record |
|---|---:|---:|---:|---:|---:|
| W26 | 1–2 | 16,384 | ~8–11 GB | ~10.5–12.5k | ~55–65% |
| B32 | 1–2 | 16,384 | ~8–11 GB | ~9.5–11.5k | ~49–60% |
| D36 | 1 | 16,384 | ~8–11.5 GB | ~8.5–10.5k | ~44–55% |
| Q36 | 1 | 16,384 | ~8.5–11.5 GB | ~8–10k | ~42–52% |

The Arctus reference (~19.2k) is a measured **short benchmark** from the existing project, not a long-run sustained rate.

## Estimated quality direction vs Arctus 350M

No 600M quality benchmark has been run yet. If trained on the same high-quality token budget and later SFT'd comparably, a strong 600M finalist is plausibly in these broad relative ranges versus Arctus 350M:

- overall capability: **+8–18%**
- reasoning: **+12–25%**
- math: **+10–22%**
- code: **+8–20%**
- German/English language: **+5–12%**
- long-context task quality: **+8–20%**, conditional on actual long-context training
- quality per parameter: **+2–8%** only if the architecture/data recipe is genuinely better; parameter scaling alone does not guarantee this

These are planning estimates, **not measured results**.

## What is checkpoint-compatible with Arctus?

Execution-only ideas can transfer conceptually without changing mathematical weights: fused QKV, fused gate/up, fused linear+cross-entropy, SDPA/Flash, compiler settings and data pipeline optimizations.

The 600M architecture itself is **not checkpoint-shape-compatible** with Arctus: width, depth, FFN dimensions and GQA geometry differ. It therefore requires new pretraining from scratch, as required by the project brief. The tokenizer can remain exactly the same.

## Prime and a future 1B model

Prime's useful principles scale — clean head_dim64, less aggressive GQA, Pre-RMSNorm/QK-Norm and fused execution — but the exact 350M Prime geometry should **not** simply be stretched to 1B. The 600M search is deliberately the evidence-generating step. Its measured width/depth/GQA Pareto frontier should determine the later ~1B search.
