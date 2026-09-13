# Neral 350M Throughput 60K V2

This branch prepares the next speed round without modifying the accepted Scale V3 model architecture or promotion floor.

## Objective

Increase **safe promoted tokens per wall-clock hour**, not merely a peak tok/s number. Any optimization that changes training data, replay semantics, trainable layers, LR policy, checkpoint identity, or quality gates is rejected as a speed comparison.

## Phase A — Pack 9

The current Scale V3 geometry uses 12 microbatches per update: 3 replay + 9 main. Pack 6 processes the main side as 6 + 3. Pack 9 can process all nine main microbatches in one physical forward/backward.

`benchmark_pack9_v2.py` compares Pack 6 and Pack 9 using the same guarded trainer, same promoted anchor, same seeds, same token budget, same replay share, same LR and same optimizer. It records median/p10/peak throughput, VRAM, frozen-weight integrity and checkpoint identity. It never promotes a checkpoint.

## Phase B — Replay fusion feasibility

`benchmark_replay_fusion_v2.py` does not alter or save weights. It compares three separate batch-4 replay forward/backward passes with one concatenated batch-12 pass on the exact same deterministic tensors.

It records:

- loss delta;
- gradient relative L2 delta;
- gradient max-absolute delta;
- median execution time;
- speedup;
- peak VRAM.

A favorable probe only authorizes a later opt-in trainer implementation and canary. It does not establish training equivalence by itself.

## Phase C — adoption gate

No speed candidate can enter Scale V3 directly. The adoption sequence is:

`bounded speed probe -> numerical A/B -> 1M own-weight canary -> full validation/dev/bilingual gate -> only then consider Scale V3 integration`

The existing promotion floor remains unchanged.

## What is intentionally unchanged

- 349,825,152 parameter Neral 350M architecture;
- 16K tokenizer and checkpoint format;
- last-four-block guarded continuation policy;
- 12 microbatches / 24,576 tokens per update;
- three replay domains and their token share;
- optimizer and loss objective;
- safe-anchor rollback semantics;
- Scale V3 autonomous control logic;
- final validation, recovery-dev and bilingual promotion gates.

## Expected interpretation

Pack 9 is the lowest-risk speed candidate because it changes only physical grouping. Replay fusion has a larger potential upside but requires stronger numerical evidence before it may touch the production trainer. Checkpoint-I/O and fused QKV/SwiGLU execution remain later independent workstreams and should be benchmarked separately so regressions can be attributed cleanly.

No 60K claim is made in advance. 60K tok/s is a stretch target; measured stable median and safe-promotion throughput are authoritative.
