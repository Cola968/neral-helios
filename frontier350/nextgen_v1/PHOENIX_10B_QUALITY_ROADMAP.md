# Phoenix 350M — Quality-first path from 2.073B to a usable chatbot

Status: execution plan for the experimental Phoenix lineage. This is not a claim that 10B tokens are required or sufficient. Promote checkpoints only on measured quality.

## Starting point

- Architecture: Neral Frontier 350M, exactly 349,825,152 parameters.
- Immutable historical anchor: 2,073,161,216 processed foundation tokens.
- Foundation target used by the repository: 10,000,000,000 tokens.
- Remaining to that ceiling: 7,926,838,784 tokens.
- Current free-answer development evidence is poor (0/8 strict on the documented short suite), so token count alone is not a success metric.
- Full-size BF16 forward/backward/AdamW/checkpoint mechanics are proven on the owner's RTX 4070 by Actions run 35884287374.
- Existing ~43.9k tok/s real-data profiling is an upper-four-block workload, not a verified 24/24-block foundation rate.

## Principle

Build three things separately and join them only after each works:

1. **Foundation capability** — broad language, knowledge, math and bilingual representations.
2. **Chat behavior** — instruction following, answer-only supervision, EOS, multi-turn state, refusal/uncertainty.
3. **Serving quality** — correct chat template, KV cache, stop handling, low repetition, deterministic serialization and a usable local UI/API.

Do not call a lower training loss a better chatbot.

## Phase 0 — freeze evaluation before more quality training

Create three disjoint evaluation lanes:

- **training validation**: token loss only; never used as a chatbot score;
- **development suite**: 100–300 DE/EN tasks across everyday knowledge, arithmetic/math, science, aviation/transport, code, instruction following, truthfulness, multi-turn and generation health; may be inspected while developing;
- **sealed holdout**: independently authored private DE/EN cases that remain unopened until a candidate has already passed the development gates.

Permanently reserve public external benchmarks already named by NIS2 (GSM8K, MBPP, CodeContests, IFEval, TruthfulQA, BoolQ, ARC). Screen exact and near duplicates out of all train sources.

## Phase 1 — find the fastest real FULL-MODEL configuration

Do not extrapolate the four-block 43.9k tok/s result.

Benchmark the exact 24/24-block model on the RTX 4070 with the immutable anchor and a disposable local output:

- BF16 + TF32;
- fused AdamW;
- persistent mmap loader;
- torch.compile(default) as baseline;
- contexts 512 first, then 1024 only if quality/data warrants it;
- microbatch sweep to the largest stable value below a conservative VRAM ceiling;
- target 16k–32k tokens per optimizer update using gradient accumulation;
- prewarm the RoPE cache before compiled training;
- compare gradient checkpointing on/off;
- include compilation, data loading, validation and checkpoint overhead in end-to-end rate.

Acceptance: same data order/seed and numerically sane loss/gradients; no OOM/thermal guard; exact resume tested; choose the highest **sustainable end-to-end** full-model rate, not the highest synthetic kernel number.

### Runtime reality

Remaining 7.927B tokens would take, before extra evaluation/checkpoint overhead:

| Sustainable full-model rate | Compute time to 10B |
| ---: | ---: |
| 15k tok/s | ~146.8 h / 6.1 d |
| 20k tok/s | ~110.1 h / 4.6 d |
| 25k tok/s | ~88.1 h / 3.7 d |
| 30k tok/s | ~73.4 h / 3.1 d |
| 44k tok/s | ~50.0 h / 2.1 d |

44k is not currently proven for full-model training. Measure first.

## Phase 2 — continue foundation training in quality-gated stages

Use the existing audited foundation mix as the initial baseline, not dialogue data:

- 52% FineWeb-Edu EN
- 20% FineWeb EN
- 13% FineWeb-2 DE
- 15% FineMath

Keep immutable source revisions, license/provenance receipts, exact document dedupe, train/validation disjointness and the locked 16,384-token BPE.

Do **not** launch one monolithic 7.927B-token job. Suggested gates:

- Anchor: 2.073B
- F1: 2.50B
- F2: 3.00B
- F3: 4.00B
- F4: 5.50B
- F5: 7.00B
- F6: 8.50B
- F7: 10.00B ceiling

Inside each stage, write restart-safe checkpoints much more frequently than the quality gates. Use validation loss and a nonsealed development suite to decide whether another large stage is justified.

Stop/adjust if:
- validation loss stalls or worsens persistently;
- generation degeneracy/repetition grows;
- German or any core domain regresses materially;
- gradient/optimizer state becomes nonfinite;
- exact-resume identity fails;
- throughput or thermals become unstable.

If quality plateaus at 6–8B, do not spend tokens merely to print "10B".

## Phase 3 — chat SFT early as a diagnostic, then final SFT after foundation

A tiny bounded SFT pilot should happen **before** committing to the entire remaining foundation budget, because the current base was not pretrained on a systematic User/Assistant distribution.

Pilot:
- reviewed DE/EN OASST-style data only;
- answer-only masked loss;
- explicit EOS;
- conversation-level dedupe/split;
- start with upper blocks only and very low LR;
- compare immutable anchor vs candidate on the development suite.

Purpose: prove that chat-format failure is trainable without destroying base capability. Do not promote this pilot.

After the best foundation checkpoint is selected, run the real chat SFT:
- several million high-quality supervised answer tokens rather than billions;
- balanced DE/EN;
- single-turn + multi-turn;
- concise and long answers;
- uncertainty/unknown handling;
- arithmetic/math explanations;
- code/science/domain tasks;
- refusal/safety examples;
- strong EOS and low-repetition examples.

Select SFT checkpoints on response correctness + instruction following + generation health, not only SFT validation loss.

## Phase 4 — preference tuning only if it solves an observed problem

Do not add DPO/ORPO automatically. Use it only if supervised training produces factually capable but poorly ranked/style-inconsistent answers. Keep preference data independent of the sealed final holdout.

## Phase 5 — make Helios a functioning product

The model is not "done" until all of these work together:

- canonical chat template used in training and inference;
- native KV-cache parity;
- EOS/role-boundary stopping;
- repetition control that does not rewrite model knowledge;
- context truncation policy;
- streaming generation;
- stable checkpoint/version manifest;
- local HTTP API + browser UI;
- deterministic smoke tests;
- latency/VRAM benchmarks;
- optional quantized inference copy, never replacing the master checkpoint.

## Promotion gates

A candidate becomes the default Helios model only when all pass:

1. checkpoint/tokenizer/data lineage and exact resume verified;
2. development-suite overall improves with no severe category collapse;
3. sealed bilingual holdout improves against the immutable anchor;
4. generation health/EOS/repetition pass;
5. NIS2 capability lane and independent public benchmark lane show no unacceptable regression;
6. multi-turn local chat works end-to-end;
7. safety/robustness and serving tests pass.

## Immediate order of execution

1. Build/freeze the private development + sealed holdout manifests.
2. Run a full-model RTX 4070 shape/compile benchmark and lock the fastest stable training configuration.
3. Prepare audited foundation shards using the existing builder and exact pinned source revisions.
4. Run a small bounded real-data foundation stage, verify exact resume and end-to-end tokens/s.
5. Run the small reviewed chat-SFT diagnostic.
6. Compare both against the immutable 2.073B anchor.
7. Continue foundation stages only while measured capability improves.
8. Final SFT on the best foundation checkpoint.
9. Final sealed evaluation and local Helios runtime qualification.

This path minimizes wasted GPU time: expensive billion-token training happens only after data, throughput, resume integrity and evaluation are already working.
