# Neral Nova 1B

Status: **research architecture only — not trained, not promoted**.

Nova 1B is the next from-scratch Neral architecture research track. It is deliberately
separate from the current Arctus 350M -> 1B-*token* continuation. The repository did not
contain a dedicated 1B-parameter architecture before this branch; the existing 600M search
only stated that its measured Pareto frontier should inform a later ~1B search.

## Core contract

Default candidate: `nova1b-m3-amw-rope`

- analytical trainable parameters: **999,873,280**
- Neral-owned Byte-BPE vocabulary: **16,384**
- model width: **1536**
- depth: **32** blocks
- sequence-mixer schedule: **Mamba-3, Mamba-3, Mamba-3, exact full attention**, repeated 8x
- Mamba-3 blocks: **24**
- exact full-attention anchors: **8**
- Mamba-3: d_state 128, expand 2, head dim 64, MIMO rank 4, RoPE fraction 0.5
- full attention: 24 Q heads / 6 KV heads, 4:1 GQA, head dim 64
- full-attention Q/K RMSNorm + dynamic per-head sigmoid output gate
- SwiGLU hidden: **3648** in every block
- Pre-RMSNorm, bias-free linear maps, zero pretraining dropout
- tied token embedding / LM head
- max architecture context contract: **16,384** tokens; long context must be trained and benchmarked before claiming effective 16K capability
- BF16 correctness baseline
- activation checkpointing enabled by default for the research implementation

The exact instantiated parameter count is fail-closed against the analytical contract.
Any upstream dependency drift must be resolved before a run is accepted.

## Why hybrid instead of simply scaling Arctus

A 1B dense full-attention transformer would increase parameter capacity but also increase
attention/KV cost and optimizer pressure on the RTX 4070. Nova instead allocates most
sequence mixing to recurrent Mamba-3 blocks and periodically inserts exact full attention.
The hypothesis is:

1. recurrent blocks provide efficient local/global state evolution and constant-size decode state;
2. exact attention anchors preserve content-addressable retrieval and token-to-token lookup;
3. periodic anchors avoid paying quadratic full-attention cost in all 32 layers;
4. the 1536-wide / 32-deep geometry keeps large GEMMs while adding substantially more transformation depth than Arctus 350M.

This is an engineering hypothesis until matched pilots exist.

## Neral-specific experiment: Adaptive Memory Weave (AMW)

Each 3x Mamba-3 segment is followed by a low-rank Neral bridge immediately before the
exact-attention anchor:

`x <- x + sigmoid(g) * up(silu(down(x)))`

with rank 128 and a learned channel-wise injection gate. The bridge adds only **394,752
parameters per cycle**, **3,158,016 total**.

The intended role is to give the exact-attention anchor an explicit learned compression /
re-expression of the preceding recurrent stream before exact retrieval. This is inspired by
recent results on hybrid attention and cross-layer routing, but **AMW is only a Neral-specific
novelty hypothesis**. This repository does not claim patent novelty or that no similar method
exists elsewhere. Promotion requires a matched no-AMW ablation.

## Three frozen architecture candidates

| Candidate | AMW | Full-attention positions | Analytical params | Purpose |
|---|---:|---|---:|---|
| `nova1b-m3-amw-rope` | yes | RoPE | **999,873,280** | default Nova hypothesis |
| `nova1b-m3-nobridge-rope` | no | RoPE | **996,715,264** | isolates AMW value |
| `nova1b-m3-amw-nope` | yes | NoPE | **999,873,280** | positional-anchor ablation |

No candidate is a winner before real measurements.

## Current research basis (September 2026)

The architecture is intentionally conservative about what is copied and aggressive about
what is tested.

- **Mamba-3** is used as the current recurrent-mixer research basis. Nova pins upstream
  `state-spaces/mamba` commit `e9594ce1c732d97440f0332fdc43170a2294dbfa` and uses it as
  a code/math dependency only. No Mamba weights are used.
- Recent hybrid-architecture work indicates that exact full attention remains important for
  long-range retrieval even when efficient sequence mixers improve optimization/efficiency.
- Recent linear-attention comparisons show architecture/optimizer trade-offs are large and
  that hybrid stacks can improve loss at a throughput cost; this motivates explicit Pareto
  measurement rather than assuming the newest mixer is automatically best.
- A NoPE-on-full-attention result in recent hybrid research is included only as a controlled
  ablation (`nova1b-m3-amw-nope`), not as a default assumption.
- Full-bandwidth latent feedback is interesting but is **not stacked into Nova v1** because it
  changes the training schedule/teacher-forcing path and would confound the first hybrid test.
- MoE is not a Nova v1 baseline because a single 12 GB GPU still bears expert parameter and
  optimizer storage plus routing overhead.

## Provenance / licensing

Nova remains a Neral model:

- no foreign weights or checkpoints;
- no foreign tokenizer vocabulary or merges;
- Neral's existing 16K tokenizer remains the tokenizer contract;
- external code dependencies are pinned, auditable dependencies only;
- upstream Mamba repository license at the pinned research revision: Apache-2.0;
- upstream copyright/license notices must be preserved when distributing dependency code.

## RTX 4070 strategy

A 1B model cannot be treated like the current 350M training job. Rough BF16 model weights
alone are about 2 GB and BF16 gradients another ~2 GB; conventional FP32 Adam moments can
consume many additional GB before activations. Therefore Nova separates architecture quality
from optimizer experiments.

Research order:

1. analytical/CPU contract CI;
2. pinned Mamba-3 runtime validation in WSL;
3. BF16 forward/backward GPU smoke at seq 256/512;
4. measured microbatch/checkpoint sweep under a hard <12 GB gate;
5. optimizer memory ablation: 8-bit AdamW control vs. Muon-family research path only after numerical checks;
6. matched 1M–5M token learning-curve pilots on identical data/seed;
7. longer 25M/100M confirmation only for Pareto survivors.

Do not quote Nova throughput, VRAM, loss, NIS2, context quality or capability as measured
until the corresponding GPU/pilot run has completed.

## Promotion gates

The default candidate must beat or match its controls on a multi-objective gate:

- validation CE / perplexity at matched tokens;
- validation CE per training FLOP;
- end-to-end training tokens/s;
- peak VRAM and OOM margin;
- gradient/loss stability;
- short-context generation health;
- 8K/16K retrieval only after explicit long-context training;
- inference/decode speed and recurrent-state/KV memory;
- no regression large enough to erase the capacity gain versus the eventual 600M/Arctus controls.

AMW is promoted only if `nova1b-m3-amw-rope` materially beats
`nova1b-m3-nobridge-rope` at matched training conditions. NoPE is promoted only if it wins
the dedicated positional ablation.

## Relationship to the existing 600M search

The existing `research/neral600-architecture-search` branch is still valuable evidence. It
contains 600M width/depth/GQA candidates but has no promoted winner. Nova does not pretend
those measurements already exist. If the 600M hardware/learning-curve search finishes first,
its measured Pareto frontier should be used to revise Nova's width/depth or control model
before any expensive Nova pretraining.

## Not started

This branch intentionally contains **no live benchmark request** and no training request.
It does not interrupt Arctus training/data work. A benchmark must be explicitly triggered
after runtime review.
