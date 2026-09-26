# Phoenix 350M — Quality-first path from 2.073B to a usable chatbot

Status: execution plan for the experimental Phoenix lineage. This is not a claim that 10B tokens are required or sufficient. Promote checkpoints only on measured quality.

## Starting point

- Architecture: Neral Frontier 350M, exactly 349,825,152 parameters.
- Immutable historical anchor: 2,073,161,216 processed foundation tokens.
- Foundation target used by the repository: 10,000,000,000 tokens.
- Remaining to that ceiling: 7,926,838,784 tokens.
- Current free-answer development evidence is poor (0/8 strict on the documented short suite), so token count alone is not a success metric.
- Full-size BF16 engineering forward/backward/AdamW/checkpoint mechanics are proven on the owner's RTX 4070 by Actions run 35884287374; this is not the precision contract for the new productive lineage.
- New productive warm-start training uses **FP32 master parameters + FP32 AdamW moments + BF16 CUDA autocast compute**.
- Actions run **35971556138** measured the exact 24/24 FP32-master compute loop on the owner's RTX 4070. Best tested shape: seq256 / microbatch4 / grad-accum1 / no gradient checkpointing / torch.compile(default), **7,453.45 tok/s**, 6,184 MiB peak reserved VRAM.
- Existing ~43.9k tok/s upper-four-block and ~18.65k tok/s BF16-master full-model engineering rates are references only, not FP32-master planning rates.

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

## Fast lane — get a usable Helios before the 10B ceiling

Same-scale public projects show that conversational behavior can be added with a much
smaller post-training phase than foundation pretraining. That is useful **research
evidence about training scale**, not permission to copy their training data.

Under Neral's current executable data policy, external-model-generated rows remain
blocked. Therefore the Phoenix fast lane is:

1. keep the immutable 2.073B anchor;
2. stage the pinned **OASST2 human-only DE/EN** source already registered by Neral;
3. require independent privacy/content/license review plus exact/near benchmark screening
   for the concrete Phoenix pack;
4. run response-only SFT with explicit EOS and a deliberately small bounded budget;
5. compare raw free answers and generation health against the immutable anchor;
6. keep foundation pretraining as a separate lineage and rerun the inexpensive SFT recipe
   on whichever foundation checkpoint eventually wins.

`HuggingFaceTB/smol-smoltalk`, `HuggingFaceTB/smoltalk`,
`HuggingFaceH4/ultrachat_200k` and
`HuggingFaceH4/ultrafeedback_binarized` remain **research references only** under the
current policy because they contain synthetic/model-generated training or preference
content. Their permissive top-level Hub licenses do not override Neral's data-origin rule.
Do not add DPO merely because another small model used it; preference tuning stays deferred
until Phoenix has an observed ranking/style failure and a separately admitted preference
source.

## Phase 1 — find the fastest real FULL-MODEL configuration

Do not extrapolate the four-block 43.9k tok/s result.

Benchmark the exact 24/24-block model on the RTX 4070 with the immutable anchor and a disposable local output:

- FP32 master parameters and FP32 AdamW moments;
- BF16 CUDA autocast compute + TF32 matmul where applicable;
- fused AdamW;
- persistent mmap loader;
- torch.compile(default) vs eager;
- start with seq=256 and the exact safe 1×4 / 2×2 / 4×1 update geometries used by the bounded refresh path;
- compare gradient checkpointing on/off;
- keep reserved VRAM below the conservative 10 GiB ceiling;
- only enlarge context/update geometry after this precision-correct baseline is stable;
- include compilation, data loading, validation and checkpoint overhead before calling any rate end-to-end.

Acceptance: same data order/seed and numerically sane loss/gradients; no OOM/thermal guard; exact resume tested; choose the highest **sustainable end-to-end** full-model rate, not the highest synthetic kernel number.

### Runtime reality

Actions run **35971556138** measured **7,453.45 tok/s** for the best tested FP32-master
compute-loop shape. This measurement excludes production mmap loading, validation,
checkpoint I/O and compile warm-up, so it is an **upper bound for current end-to-end
throughput**, not an ETA.

At that measured compute-loop rate, the remaining 7,926,838,784 tokens from the immutable
2.073B anchor to the 10B ceiling require about **295.4 compute-hours / 12.3 days** before
any evaluation, checkpointing, data-loading or interruption overhead.

Approximate pure-compute cost between named quality gates at the same rate:

| Gate | New tokens from previous gate | Pure compute |
| ---: | ---: | ---: |
| 2.50B | 426.84M | ~15.9 h |
| 3.00B | 500M | ~18.6 h |
| 4.00B | 1.0B | ~37.3 h |
| 5.50B | 1.5B | ~55.9 h |
| 7.00B | 1.5B | ~55.9 h |
| 8.50B | 1.5B | ~55.9 h |
| 10.00B ceiling | 1.5B | ~55.9 h |

Therefore Phoenix should **not** commit to a blind 10B continuation. First determine whether
the 2.50B and 3.00B gates produce meaningful development-suite and generation-health gains.
If not, spend compute on data/SFT/architecture corrections instead of more tokens.

## RTX 4070 performance evidence and precision correction

On 23 September, BF16-master engineering probes reached up to ~18.65k tok/s at
seq256/micro16/compile. Those measurements were useful for proving compute shape and compile
benefit, but **they are no longer the planning rate for productive Phoenix training**.

As of 24 September, the new-lineage precision contract is:

- FP32 serialized master parameters;
- FP32 AdamW first/second moments;
- BF16 CUDA autocast for forward/backward compute;
- exact resume must preserve this precision identity.

Actions run **35971556138** completed that matrix successfully. Results:

- 1×4, gradient checkpointing, compile: **916 tok/s**, 6,660 MiB reserved;
- 2×2, gradient checkpointing, compile: **1,584 tok/s**, 6,678 MiB reserved;
- 4×1, gradient checkpointing, compile: **3,153 tok/s**, 5,906 MiB reserved;
- 2×2, no checkpointing, compile: **5,265 tok/s**, 7,352 MiB reserved;
- 4×1, no checkpointing, compile: **7,453 tok/s**, 6,184 MiB reserved;
- 4×1, no checkpointing, eager: **4,251 tok/s**, 6,844 MiB reserved.

The current compute-loop winner is therefore **seq256 / micro4 / accum1 / no-GC /
compile(default)**. Compile warm-up was expensive (~89 s for the no-GC compiled shapes), so
short diagnostic runs can still be faster in eager mode even though sustained compiled updates
are faster. The next performance claim must come from the real mmap corpus + validation +
checkpoint loop; do not call 7.45k tok/s end-to-end yet.

## Phase 2 — continue foundation training in quality-gated stages

The production builder now has an explicit **policy-approved baseline v2**:

- 61% FineWeb-Edu EN
- 24% FineWeb EN
- 15% FineWeb-2 DE
- 0% FineMath

The smaller development refresh intentionally uses 55% FineWeb-Edu / 15% FineWeb / 30%
FineWeb-2 DE to stress bilingual preservation. Both recipes call the executable source policy
before any network/data side effect.

`HuggingFaceTB/finemath` remains a separately pinned candidate. It can be introduced only
after candidate admission (immutable revision/file hashes, deterministic cleaning provenance,
reserved-benchmark contamination screening) and an explicit policy/recipe change. Do not treat
`source_lock.json` as authorization; a lock proves identity, not permission.

For every admitted recipe keep immutable source revisions, license/provenance receipts,
exact document dedupe, train/validation disjointness and the locked 16,384-token BPE.

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
