# Neral 350M Scale V6 — StreamGuard

Status: design/implementation target only. This file does **not** activate V6 and must not affect the currently running V5 job.

## Objective

Optimize **accepted tokens per wall-clock second**, not a short raw tok/s peak.

Targets after validation on the RTX 4070:

- raw replay-only training: >= 50k tok/s target, 60k stretch goal
- effective throughput including 5M health checks/checkpointing: >= 40k tok/s, 45k+ preferred
- typical trainer reserved VRAM: <= 8.5 GiB, hard operating target <= 9.0 GiB
- free-VRAM safety margin while training: >= 1.5 GiB preferred
- logical rollback bound: <= 5M tokens
- accepted 5M stages: >= 95% over a 100M validation run
- strict SAFE/release gate unchanged

## Core architecture

### 1. Persistent 50M resident segment

Load model, optimizer, mmap pools and compile graphs once. Train ten logical 5M transactions inside the same process. Do not restart/recompile at every 5M boundary.

A logical 5M transaction still has its own health decision and rollback point. Every 50M, or earlier if the memory watchdog requests it, close the resident segment and restart from the latest committed point. This bounds allocator fragmentation and long-process failure risk while preserving most compile/startup savings.

### 2. Transactional 5M commits

Before a logical transaction, capture:

- sampler RNG state
- torch CPU/CUDA RNG state
- batch-chain state
- optimizer step metadata
- latest committed trainable-state checkpoint identity

Train exactly 5M tokens. Run the health screen. If healthy, atomically publish the new trainable-state snapshot. If unhealthy, restore the previous committed trainable state **in-process** and adjust the next recipe. No model reload and no compile restart are required.

### 3. Flat trainable-state snapshots, not a delta chain

A 5M recovery snapshot contains the complete state required for the trainable last four blocks plus optimizer/RNG/provenance and references an immutable full-base checkpoint by SHA-256.

Important: each 5M snapshot depends only on one full base, never on earlier 5M deltas. Therefore corruption of one snapshot cannot break a long delta chain.

Create a complete full checkpoint at least every 50M and on every strict SAFE promotion. Keep two full generations plus the newest three 5M transaction snapshots.

### 4. Memory watchdog and bounded resident lifetime

Record the post-compile steady-state reserved VRAM baseline. Request a clean segment restart from the latest committed 5M point when any condition is met:

- segment reaches 50M tokens
- reserved VRAM grows >= 768 MiB above the steady-state baseline
- free VRAM falls below 1536 MiB outside a known temporary evaluation allocation
- allocator OOM/retry occurs twice in the same segment

A watchdog restart is not a training rollback: it resumes the newest committed 5M transaction.

### 5. OOM-safe exact-update retry

Sampling is separated from GPU execution. Prepare an update's token rows in pinned CPU buffers first, then snapshot its RNG/batch-chain metadata. If a physical execution profile OOMs before optimizer.step:

1. zero gradients and drop transient tensors
2. empty only the CUDA cache, never model/optimizer state
3. restore the pre-update logical metadata
4. reuse the **same prepared token rows**
5. retry with the next lower-memory physical profile

The data sequence, token counts, LR, replay weighting and logical objective do not change.

Physical profiles are fixed for a resident segment to avoid compile thrashing:

- P48: one fused physical batch of 48 sequences
- P32: 32 + 16 split, same logical update
- P24: 24 + 24 split, same logical update

Choose once at segment startup from benchmark + VRAM headroom. Do not switch upward mid-segment. An OOM may switch downward and force a clean graph rebuild once.

### 6. Frozen-head chunked CE experiment

The LM head/tied embedding is frozen in continuation training. The full 48x512x16384 logits tensor is a major avoidable memory object.

Implement a custom test path that computes cross-entropy over vocabulary chunks and recomputes chunk logits in backward, producing gradients only for hidden states. Do **not** enable it in production until equivalence gates pass.

Required equivalence on at least 100 fixed batches against standard PyTorch CE:

- absolute scalar loss difference <= 2e-4
- gradient cosine similarity >= 0.99999
- relative gradient-norm difference <= 5e-4
- identical target masking/token accounting
- no quality regression in a 25M continuation A/B run

If these gates fail, keep standard CE and use P32/P24 for VRAM control instead. Correctness wins over memory savings.

### 7. In-process micro-health, isolated full audit

The cheap 5M health screen runs in the resident process to avoid reloads. It must:

- snapshot CPU/CUDA RNG
- call model.eval()
- use torch.inference_mode()
- use fixed deterministic case order/seeds
- restore model.train() and all RNG states before training resumes

Cache SAFE baseline smoke/validation results by SAFE checkpoint SHA and rotation; never recompute an unchanged SAFE baseline every 5M.

The expensive 50M strict SAFE audit remains an independent evaluator process. This intentionally preserves an isolation boundary for release decisions even though it costs some wall time.

### 8. Preventive retention controller with hysteresis

Do not react to one noisy benchmark flip. Use only already-produced 5M health evidence.

Pressure may rise by at most one level per 5M stage. Relax pressure by one level only after three consecutive clearly healthy accepted stages.

Escalation evidence includes:

- validation delta trend worsening
- regressions minus gains > 0
- negative smoke margin trend
- recent rejected transaction

Pressure changes preserve the existing V5 semantics: lower LR and higher replay boost. Projected rescue remains exceptional, not the steady-state path.

This avoids the oscillation failure mode of rapidly raising/lowering replay and LR.

### 9. Double-buffered input path

Use two persistent pinned CPU buffers and two GPU input buffers. While GPU computes update N, fill the CPU buffer for N+1. Transfer on a dedicated CUDA stream and synchronize only on buffer consumption.

The prepared-update object also enables exact OOM retry without resampling.

### 10. Telemetry semantics

Report both:

- raw_tps: accepted training tokens / active GPU training time
- effective_tps: newly committed continuity tokens / total wall time since segment start

Also report compile time, health-eval time, checkpoint-copy/write time, audit time and rejected-token cost separately. A 60k raw peak is not considered a success if effective throughput remains low.

## Acceptance sequence

V6 must not replace V5 immediately.

1. Unit/equivalence tests for prepared-update retry and trainable-state reconstruction.
2. 5M dry validation from a copied continuity anchor; compare token counts, batch chain and endpoint identity rules.
3. 25M A/B against V5 with the same anchor and data contract.
4. 100M endurance test: verify >=95% accepted stages, <=5M rollback, no monotonic VRAM growth, and effective throughput target.
5. Only then use V6 for 2B -> 10B.

## Non-goals / explicitly rejected shortcuts

- no FP8 or 8-bit optimizer until separately quality-validated
- no weaker SAFE gate
- no larger rollback window to gain speed
- no silent data/resampling change on fallback
- no long chained delta checkpoints
- no dynamic shape/profile hopping on every update
- no claim of 60k until measured on the actual RTX 4070 runner
