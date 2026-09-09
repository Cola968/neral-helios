# Next GPU phases — prepared while RTX 4070 is offline

All values in this document are plans unless explicitly labelled **measured historical evidence**. No GPU work was executed for this file.

## Gate 0 — first job in the next GPU window

**Run the complete frozen NIS2 lineage benchmark to completion before producing new promoted weights.**

Reason: the prior NIS2 run reached B2 but was interrupted during RawMath by self-hosted runner shutdown. A complete anchor profile is required for later Pareto decisions. Missing NIS2 components may not be imputed.

Expected action order:

1. validate local checkpoint SHA and frozen-suite hashes;
2. complete B2/selected-anchor NIS2 + RawMath + long-context + targeted + generation evidence;
3. persist reports only;
4. only then choose whether the next weight-producing job is I1 resume or an isolated data/math pilot.

## Gate 1 — fair hardware benchmark

Run the prepared reproducible matrix before ORION architecture selection:

- Neral 50M Legacy
- Neral 50M Fused MLP
- ORION A3
- ORION A4
- seq 256 / 512 / 1024 / 2048 / 4096
- seq 8192 only where the real allocation fits
- comparable token budget per update where methodologically valid
- BF16
- checkpointing off/on for ORION at long sequence lengths

Measure synchronized forward, forward+backward, optimizer, tokens/s, tokens/hour, peak allocated/reserved VRAM and OOM. GPU-utilization sampling is supplemental and must state sampling source/cadence.

No architecture promotion from static calculations alone.

---

# Scenario A — improve the current Neral 50M lineage

## Objective

Improve the verified 50M anchor specifically on:

1. EOS / reliable stop behaviour;
2. repetition;
3. exact math generation;
4. transfer and OOD math;
5. instruction/multi-turn quality;

while protecting DE/EN language modelling, code, context and forgetting metrics.

## A0: resume discipline

- Resume only from a checksum-verified local checkpoint whose code/config/data-manifest fingerprints match the intended phase.
- Never infer progress from a directory name.
- Existing interrupted I1 layout remains authoritative until a clean phase boundary; do not silently convert optimizer parameter layouts mid-phase.
- At the first clean phase boundary, adopt the stronger atomic/checksum/`last_good` envelope or an explicit compatibility adapter.

## A1: complete B3a

Current configured target: 300M cumulative byte tokens.

Existing I1 configuration:

- seq 512
- batch 16
- 8192 tokens/update
- peak LR `6e-5`
- floor LR `6e-6`
- warmup 2%
- no training-block replay
- validation initially every 100 updates and then every 250 according to the campaign controller
- selection: lowest new B3 macro CE subject to frozen-B2 regression guard

**Measured historical 50M hardware evidence:** approximately 72k–76.6k optimizer tokens/s has been observed on the RTX 4070 for comparable 50M BF16 training configurations. This is not a prediction for every future sequence length or fastpath.

The committed B3a receipt shows 105.68M cumulative tokens, so that receipt alone leaves ~194.3M tokens to the 300M target. Pure optimizer time at the historical 72k–76.6k range would be roughly 42–45 minutes, before validation/checkpoint/data overhead. The local runtime may have advanced beyond that committed receipt; therefore this is a conservative receipt-based estimate, not a remaining-time promise.

## A2: B3b

Configured target: 300M -> 500M cumulative.

- seq 1024
- batch 8
- 8192 tokens/update
- peak LR `3e-5`
- floor LR `3e-6`
- same freshness/document-boundary discipline

Do **not** reuse the seq-512 throughput as a claimed seq-1024 speed. The first few hundred measured B3b updates should establish the real throughput and memory baseline before any runtime estimate is updated.

## A3: Math repair curriculum

After the selected pretraining anchor is frozen:

Planning mix for a targeted repair stage:

- 35% general-language replay
- 20% reviewed human SFT replay
- 15% Math ID
- 20% Math transfer
- 10% Math OOD + physics-math

This is a curriculum target, not a requirement to replace the existing M1/M2 implementation. Existing RawMath/M2 transfer machinery should be reused.

Suggested optimization envelope for the first targeted experiment:

- start from current M1/M2 learning-rate scale, not from pretraining LR;
- BF16;
- gradient clipping 1.0;
- short, transactional blocks;
- development selection must be benchmark-blind;
- frozen ID/transfer/OOD tests only after the candidate is selected;
- release gate requires ID gain **and** transfer/OOD gain, with no hard regression blocker.

## A4: chat/SFT repair

Use only screened human OASST2 rows plus controlled LM replay. Keep EOS tokens structurally correct; never fabricate EOS on truncated answers.

Existing I1 instruction design already provides the primary starting point:

- up to 20k contamination-screened examples;
- human-only OASST2 lane;
- assistant-targeted loss plus base-language replay;
- max 4000 steps;
- post-selection frozen benchmark.

Do not promote on CE alone. EOS, repetition, instruction following, dialogue state, truthfulness and robustness all remain release gates.

---

# Scenario B — start ORION ~100M from scratch

## Architecture selection first

Candidate set is A3 vs A4. A3 is the static throughput-first hypothesis; A4 is the depth/balance comparator. The fair RTX benchmark decides which architecture enters pretraining.

## Initial token budget

**Research target:** 2B byte tokens for the first serious ORION campaign, expandable toward 5B only when held-out slopes and NIS2-compatible downstream evidence support continued compute.

Do not treat 2B as a magic adequacy threshold. It is the first staged budget for a ~100M from-scratch byte-level model on the available hardware.

## Proposed sequence curriculum

- S0: 0–50M tokens, seq 256 — implementation/numerical smoke, not a quality stage.
- S1: 50–500M, seq 512 — broad language acquisition.
- S2: 500M–1.25B, seq 1024 — continued general pretraining.
- S3: 1.25B–1.8B, seq 2048 — longer-context exposure.
- S4: 1.8B–2.0B, mixed 2K/4K with a small 8K lane only if measured VRAM/throughput justify it.

The exact curriculum may be changed by measured loss/throughput evidence; sequence exposure should not be increased merely to advertise a context length.

## Proposed optimizer contract

Starting research configuration, to be A/B checked during S0/S1:

- AdamW
- BF16
- peak LR `2.5e-4`
- cosine decay
- LR floor 10% of peak
- warmup 2% of the current stage
- weight decay `0.10`
- gradient clipping `1.0`
- dropout `0.0` unless quality evidence later supports nonzero dropout
- deterministic seed recorded in checkpoint and data manifests
- tied byte embedding/output head
- fused QKV + fused gate/up
- GQA
- Q/K RMSNorm
- activation checkpointing only when its measured memory/throughput tradeoff is favourable

These are planned hyperparameters, not proven optimum values.

## Data mix before quarantined math/code sources are admitted

Initial safe mix from currently approved sources:

- 40% FineWeb2 German
- 25% FineWeb English
- 35% FineWeb-Edu English

No FineMath/OpenWebMath/Stack-Edu/CodeParrot contribution until their admission gates pass.

After a math source is admitted, a later staged mix may reserve roughly 5–10% for audited math-web material and 3–5% for project-generated transfer/OOD math while reducing general web proportionally. Human OASST2 belongs in post-pretraining SFT, not bulk base pretraining.

## Sharding / resume

- immutable HF revision per source;
- quality filter before tokenization;
- exact dedup + MinHash/LSH near-dedup;
- split before packing;
- document boundaries preserved;
- uint16 Neral byte tokens;
- deterministic shuffling/mixing seed;
- source and shard SHA-256;
- data cursor in checkpoint;
- optimizer/scheduler/RNG state;
- architecture/config/manifest/source-commit hashes;
- atomic save + checksum + last-good pointer;
- no checkpoint/corpus upload to GitHub artifacts.

## Validation cadence

Use token-based rather than arbitrary wall-clock cadence:

- dense early validation during S0/S1 while failures are cheap;
- then periodic held-out DE/EN CE/BPB at fixed cumulative-token intervals;
- full model-only frozen benchmark only at explicit promotion milestones, not every checkpoint;
- external public holdouts stay evaluation-only and never drive gradient/checkpoint selection.

## VRAM and runtime

Static BF16 weights are ~190.7–190.9 MiB for A3/A4. KV-cache calculations and analytical training-state budgets exist in `orion_static_metrics.py`, but **actual training VRAM and tokens/s remain unmeasured** for ORION.

Therefore no ORION training-time forecast is considered valid until the A3/A4 BF16 RTX matrix runs. Once measured tokens/s exists, token-budget time should be reported as a range including validation/checkpoint/data overhead, not as a single artificial precision number.

---

# Prepared execution order

1. Complete frozen NIS2 anchor benchmark.
2. Run 50M Legacy/Fused + ORION A3/A4 performance matrix.
3. Resume/finish I1 B3a from verified local state.
4. Complete B3b if validation/forgetting gates remain healthy.
5. Run targeted Math/EOS repair using existing M1/M2 machinery plus the new transfer/OOD release gate.
6. Complete screened OASST2 SFT and frozen chatbot readiness evidence.
7. Select A3/A4 from real RTX evidence.
8. Freeze ORION source manifests and begin S0/S1 from scratch.
