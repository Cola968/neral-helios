# Nova 1B — optimizer and memory plan

Status: **planned / unmeasured**. Architecture and optimizer gains must be attributed separately.

## Why the Arctus optimizer recipe cannot simply be copied

Nova is ~1B trainable parameters. Approximate persistent tensor memory before activations is
already substantial:

- BF16 parameters: ~2.0 GB decimal (~1.86 GiB)
- BF16 gradients: ~2.0 GB decimal (~1.86 GiB)
- two FP32 Adam moments: ~8.0 GB decimal (~7.45 GiB)

That combination alone is near/exceeds the practical 12 GB RTX 4070 budget once allocator,
Mamba state, logits and activations are included. Full-precision AdamW is therefore not the
assumed local baseline.

## Research order

### A. Architecture feasibility (no optimizer)

Run `bench1b.py` first with BF16 + full block checkpointing. It measures forward/backward
throughput and peak VRAM without confusing architecture cost with optimizer state.

### B. Memory-feasible control optimizer

The first optimizer-inclusive pilot should use a well-tested 8-bit AdamW implementation if
it is stable in the dedicated WSL runtime. This is the control path, not a quality claim.
Parameters that require higher precision can remain in FP32 state if needed.

### C. Muon-family ablation

Recent 2026 studies make Muon-family methods a high-interest candidate for final validation
loss / optimization efficiency. Nova will not assume that result transfers to this hybrid.
A Muon or memory-reduced Muon path is admitted only after:

- finite-loss and finite-gradient smoke;
- optimizer-state VRAM measurement;
- one-step numerical sanity checks;
- identical seed/data/token budget against the 8-bit AdamW control;
- no hidden change in weight decay or learning-rate semantics.

Embedding, normalization and non-matrix parameters may need a conventional optimizer while
large 2-D matrices use Muon; that split must be explicit in reports.

## Initial training-memory controls

- BF16 weights/activations as correctness baseline
- block activation checkpointing on by default
- microbatch 1 at seq 512 first
- gradient accumulation to recover effective tokens/update
- fused kernels where mathematically equivalent
- no optimizer offload unless GPU-resident memory-efficient optimizers fail
- no FP8 baseline; FP8 is a later equivalence/speed experiment only
- no MoE baseline

## Promotion rule

A faster optimizer recipe is rejected if it materially worsens validation loss at matched
training tokens. A lower-loss recipe is rejected for the local product path if it cannot fit
with safe VRAM margin or is operationally fragile on the RTX 4070.
