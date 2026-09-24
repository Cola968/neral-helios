# Helios 72h chatbot sprint — decision contract

## Objective

Produce a locally usable German/English Helios chatbot from Neral's own 349,825,152-parameter
weights within 72 hours. This sprint optimizes for **observable conversational usefulness**,
not for reaching the 10B foundation-token ceiling.

The immutable historical anchor remains:
- checkpoint SHA-256: `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`
- historical processed tokens: 2,073,161,216
- tokenizer SHA-256: `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`

## Current fast path

1. Prepare 12,000 strict-clean SFT examples.
2. Use 45% human OASST2 DE/EN (Apache-2.0) within measured clean capacity.
3. Use 55% deterministic project-authored German/math/code/reasoning examples; no external-model answers.
4. Train upper 8 transformer blocks + final norm only.
5. Keep FP32 master weights and FP32 AdamW moments; use BF16 only for compute.
6. Use assistant-only loss with explicit EOS weighting.
7. Prefer 85% SFT + 15% historical-distribution replay when the verified local replay corpus exists.
8. Evaluate at staged stops 600 → 1400 → 2000. Stop immediately when the public-development chat MVP gate passes.
9. Preserve compact exact resume state at each evaluation boundary.

## Why not 10B first?

The historical model is known to be poor in free-answer chat despite having learned some
foundation structure. A multi-billion-token continuation would consume the entire sprint
before testing the highest-probability failure mode: chat-format/instruction mismatch.
Chat SFT is therefore tested first.

## Evidence hierarchy

A lower SFT validation loss is necessary but not sufficient.

The sprint candidate must also be compared against the immutable base on:
- direct DE/EN instructions,
- simple arithmetic and factual answers,
- short science facts,
- two multi-turn memory cases,
- non-empty generation,
- repetition health.

The current 10-case suite is a **development gate**, not a release benchmark. Passing it
means "continue to interactive local testing", never "quality verified".

## Decision tree

### A — 600-step candidate passes chat MVP gate
Stop SFT immediately. Launch the local browser chatbot and perform manual DE/EN multi-turn
testing. Do not spend GPU time on 1400/2000 merely because those stages exist.

### B — 600 fails, 1400 passes
Keep the 1400-stage best checkpoint. Do not continue to 2000.

### C — 1400 fails, 2000 passes
Keep the 2000-stage best checkpoint. Move to interactive testing.

### D — validation improves but 2000 chat gate still fails
Do **not** add random training steps. Inspect raw A/B outputs and classify the dominant
failure:
- format/EOS failure → adjust chat SFT/EOS recipe;
- repetition/degeneration → lower LR or narrower trainable subset;
- factual ignorance with otherwise good chat → use a bounded foundation refresh;
- catastrophic forgetting → increase verified foundation replay share.

Only one trajectory variable changes per next experiment.

### E — CUDA OOM / VRAM ceiling
Retry the same data and optimizer trajectory with batch 2 before changing model blocks.
Do not introduce gradient checkpointing unless batch 2 still fails; partial-block SFT
should not need full-model checkpointing on a 12GB RTX 4070.

### F — thermal pressure
Use the existing bounded cooldown/hysteresis path. Do not raise the 78 C safety trigger
merely to increase throughput.

## Optional day-2 foundation repair

Only if the candidate chats coherently but lacks basic knowledge/reasoning:
- run a bounded approved-source foundation refresh, initially 100M tokens;
- compare against the immutable base on foundation development metrics;
- apply the **same winning chat SFT recipe** to the better foundation checkpoint;
- compare both chat candidates.

At the measured ~7,453 tok/s FP32-master full-model compute-loop rate, 100M tokens is about
3.7 pure-compute hours before validation/checkpoint/cooldown overhead. Therefore 100M is a
reasonable first repair budget; multi-billion-token training is not a 72h default.

## Day-3 product gate

A sprint candidate can be called a **local MVP chatbot** only after:
- automated development chat gate passes or manual testing provides stronger evidence;
- at least 20 manually inspected DE/EN prompts across multi-turn, concise answer, math,
  science/code and uncertainty behavior;
- EOS/repetition behavior is acceptable;
- the historical source checkpoint remains unchanged;
- the local launcher uses the exact candidate SHA.

This sprint never changes the project's rule that release/promotion requires stronger
independent holdout and safety evidence.
