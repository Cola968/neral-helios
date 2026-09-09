# Neral/Helios CPU sprint audit — 2026-09-09

## Evidence boundary

This audit was produced while the local RTX 4070 was intentionally unavailable. It authorizes no CUDA work. Statements below distinguish repository implementation, committed execution evidence and future plans. A workflow being present is not evidence that its training run completed.

## 1. Current 50M lineage

### Stable anchor

The strongest checkpoint that is both locally referenced by the current campaign and suitable as a conservative anchor is the selected B2 checkpoint `runs/neral50m_b2_100m/best_validation.pt`. The B3 source receipt records its exact SHA-256 as `1d7da2b62c82159c720e52ab9809e53d484613dcf05fb01d8f77f0235c8105b6` and exact cumulative token count as **99,250,176 byte tokens**. The directory name contains `100m`, but the exact receipt value is authoritative.

### I1

`run_50m_i1.py` is the current intensive campaign controller. Its current v4 design is:

- B3a: B2 -> 300M cumulative, seq 512, batch 16;
- B3b: -> 500M cumulative, seq 1024, batch 8;
- RawMath: 30M mixed tokens with 35% math / 65% language replay;
- Instruction: contamination-screened OASST2-centred SFT, human-only OASST2 fraction and language replay;
- frozen benchmark only after the selected candidate.

Committed B3a evidence proves a valid training start and records step 785 at 105,680,896 cumulative tokens. It does **not** prove B3a completion. Any later local progress that is not represented by a committed receipt must remain runtime-local evidence until exported as a small report.

Status: **implemented + partially executed, incomplete, not promoted**.

### I2

The I2 E2 tri-loss weakness-repair workflow is implemented. It explicitly requires a **completed I1 instruction checkpoint** and then combines SFT, math and LM replay with EOS weighting and a hard post-training weakness gate. Therefore it is not currently eligible to run from the incomplete I1 B3a state.

Status: **implemented/prepared, blocked on completed I1 evidence**.

### M1

M1 generalization is implemented as a guarded automatic lane after the stage-ablation workflow. It trains ID math + transfer math + SFT + LM replay and validates a selected source by SHA-256.

Status: **implemented/prepared; this audit does not claim a completed promoted M1 checkpoint**.

### M2 / M2.1

`m2_transactional50.py`, `train_m2_pareto50.py` and their tests/workflows implement benchmark-blind transactional development selection. Existing M2 logic already requires transfer progress, not just ID math improvement. M2.1 adds stricter health/safety margins around UTF-8, German preservation, repetition and EOS.

Status: **controller/gates implemented; no new M2 weight result is claimed in this CPU sprint**.

### H1

`run_50m_h1_intensive.py`, `h1_gate50.py` and workflows prepare the 1B-cumulative-token research lane. H1 is evidence-gated and must not be started merely to consume more tokens.

Status: **prepared research campaign, not evidence of completed 1B training**.

## 2. NIS 1.0 / NIS 2.0

NIS 1.0 remains historical project-local evidence. NIS 2.0 is the current scalar capability score and remains explicitly model-only: no Helios tools, SOLVE, retrieval or external models.

Current NIS2 weights remain:

- Language/knowledge 10%
- Math generalization 15%
- Code 10%
- Science 8%
- Reasoning/compositional 12%
- Bilingual/translation 5%
- Context/dialogue 12%
- Instruction following 10%
- Truthfulness/robustness 8%
- Generation health 10%

The complete NIS2 lineage workflow was interrupted by self-hosted runner shutdown while evaluating B2 RawMath. The partial run is useful evidence, but a missing component must never be imputed into a full NIS2 score.

Recommendation: **freeze NIS2.0 weights for historical comparability**. Add hard profile/Pareto release gates outside the scalar score rather than repeatedly redefining the score when a weakness is discovered.

## 3. Existing promotion layers

The repository already contains several complementary gates:

- validation/progress gates for B3/I1;
- generation-health / EOS / repetition gates;
- RawMath progress and transfer gates;
- long-context gate;
- M1 source selection and math-generalization gate;
- M2 benchmark-blind Pareto development controller;
- I2 weakness-repair gate;
- H1 gate;
- external public-holdout lane;
- NIS/NIS2 scorecards.

This sprint adds `promotion_gate_v3.py` as a **release-level hard Pareto layer**, not a replacement for M2. Missing evidence fails closed. A math-targeted candidate must improve ID math and transfer or OOD, while respecting regression limits for code, context, reasoning, language-model loss, instruction following, generation health and forgetting.

## 4. Main unresolved model weaknesses

Priority order from current evidence and gate design:

1. **EOS / reliable termination** — still a release blocker.
2. **Math exact generation and math transfer/OOD** — pair preferences are insufficient evidence.
3. **Repetition / low-entropy degeneration** — improved from Pilot A but must stay below release thresholds.
4. **Instruction following and multi-turn chat** — requires completed SFT and frozen NIS2 evidence.
5. **Code generation** — needs complete model-only generation evidence, not preference-only success.
6. **Forgetting across math/chat repairs** — every targeted intervention needs LM/DE/EN preservation.
7. **4K/8K context quality** — supported architecturally in parts of the lineage but robust trained long-context behaviour remains an evaluation question.

## 5. ORION / Helix 100M status

### Older Helix candidates

- A1: 99,783,168 params; 32 layers; d=512; 16Q/4KV; head 32; FFN 1600; 8K.
- A2: 100,470,912 params; 24 layers; d=640; 10Q/2KV; head 64; FFN 1664; 8K.

### ORION candidates

- A3: 100,002,304 params; 14 layers; d=768; 12Q/2KV; head 64; FFN 2496; 8K.
- A4: 100,103,936 params; 16 layers; d=768; 12Q/2KV; head 64; FFN 2112; 8K.

ORION adds fused QKV, fused gate/up SwiGLU, Q/K RMSNorm, RoPE, GQA, compact KV cache, tied byte embeddings, BF16-compatible tensors and optional activation checkpointing. A3/A4 are **untrained architecture candidates**, so they have no NIS or quality score.

Static CPU accounting is isolated in PR #87. A3 remains the throughput-first hypothesis because it has the shallowest ORION stack and 56 structural major Linear calls/forward versus A4's 64 and Helix A2's 168. This is a calculated structural property, not measured RTX speed.

## 6. Hugging Face data audit

The connected Hugging Face account is authenticated but not PRO, so HF Jobs are unavailable. Hugging Face was therefore used for live Hub metadata/source inspection; GitHub-hosted runners remain the CPU execution lane.

### Approved by current project policy

- `HuggingFaceFW/fineweb-2`: multilingual general web, primary German lane, ODC-By-1.0.
- `HuggingFaceFW/fineweb`: general English web, ODC-By-1.0.
- `HuggingFaceFW/fineweb-edu`: educational English web, ODC-By-1.0.
- `OpenAssistant/oasst2`: Apache-2.0; only reviewed human rows are admitted for Neral SFT.

### Quarantine / blocked pending audit

- `HuggingFaceTB/finemath`: useful high-priority math source but blocked pending deep benchmark-contamination review.
- `open-web-math/open-web-math`: math candidate; current Hub inspection did not provide a sufficiently explicit commercial-license basis, so blocked pending license + contamination review.
- `HuggingFaceTB/stack-edu`: strong code candidate but file-level/mixed source license provenance and benchmark contamination need resolution.
- `codeparrot/codeparrot-clean`: Python candidate; blocked pending license/provenance and code-benchmark contamination checks.
- `allenai/dolmino-mix-1124`: mixed-component provenance/licenses; mixture-design reference rather than a single admitted source.
- `HuggingFaceTB/smollm-corpus`: blocked by current Neral synthetic/mixed-data policy.

### Candidate only

- `allenai/dolma`: ODC-By metadata, huge English corpus; subsets/provenance/contamination still need an explicit admission decision.
- `wikimedia/wikipedia`: useful encyclopedic/science anchor; release attribution/share-alike handling must be reviewed before broad commercial use in the final corpus policy.

The machine-readable decision table is `hf_dataset_manifest_v3.json`. The authoritative code policy remains `hf_source_policy50.py`; unknown or quarantined sources fail closed.

## 7. Data pipeline status

Already present before this sprint:

- immutable source revision / manifest infrastructure;
- streaming Hugging Face preparation;
- byte-token uint16 sharding;
- exact SHA-based dedup;
- document-boundary preservation;
- quality/PII/secret filters;
- source hashes and candidate-only contracts;
- multi-billion scale plan (1B/2B/5B and later 10B+);
- external public benchmark holdout registry.

Added in this sprint:

- `data_quality_v3.py`: deterministic CPU text-quality assessment plus disk-backed MinHash/LSH near-dedup lane;
- `holdout_registry_v3.py`: one fail-closed internal+public holdout registry with content hashes over frozen benchmark definitions;
- `math_curriculum_v3.py`: explicit 9-level ID -> transfer -> OOD -> physics-math curriculum contract;
- `promotion_gate_v3.py`: release-level hard Pareto gate.

Near-dedup is now an implemented component but is **not yet claimed as wired through every legacy feeder**. Final multi-source shard builders should invoke it before admission.

## 8. Resume/checkpoint safety

Existing campaigns already preserve substantial resume metadata. The ORION CPU branch additionally introduces `checkpoint_io_v3.py` as a shared future contract with atomic same-directory temp+fsync+rename, SHA-256 sidecar, `last_good_checkpoint`, optimizer/scheduler/RNG state, token/step/curriculum/data cursor and config/architecture/manifest/source hashes.

It deliberately does not silently rewrite interrupted legacy I1 checkpoints. Migration must happen only at a clean phase boundary or through an explicit compatibility adapter.

## 9. Relevant open PR topology

- #86: mobile/hosted policy-safe feeder + isolated 50M data pilot — active data/chat base.
- #85: Neral 50M V3 GPU fastpath candidates — unmeasured until RTX returns.
- #84: ORION/50M fused architecture/perf sprint — open and stale/conflicting against main.
- #87: CPU ORION static accounting + crash-safe resume contract — stacked on #84.
- #67: older Helix A2 comparison — historical architecture branch, not current preferred design.
- #83: roadmap docs — stale relative to current ORION/data work.

Do not merge overlapping architecture branches blindly. Consolidation should happen after CPU contracts pass and before the next measured hardware selection.

## 10. Priority list

1. Complete CPU CI for the new ORION/data governance contracts.
2. Finish the interrupted complete NIS2 anchor benchmark when RTX returns.
3. Run the fair 50M legacy/fused/A3/A4 BF16 hardware matrix.
4. Resume I1 only from a verified local checkpoint/receipt; complete B3a/B3b evidence before RawMath/SFT.
5. Keep FineMath/OpenWebMath/Stack-Edu quarantined until audit evidence is green.
6. Integrate MinHash/LSH into final multi-source sharding path.
7. Apply the release Pareto gate only after complete comparable frozen evidence exists.
8. Start ORION from scratch only after A3/A4 hardware selection and source manifests are frozen.
