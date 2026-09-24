# Helios Fast Quality Loop — 24 September 2026

Status: experimental execution contract for Phoenix/Arctus 350M. This document optimizes for the fastest measured route to a useful own-model chatbot. It does not treat token count, loss alone, or test count as chatbot quality.

## Objective

Produce a locally usable DE/EN Helios chatbot from the own 349,825,152-parameter Neral/Arctus lineage as quickly as possible while keeping the immutable 2,073,161,216-token anchor recoverable.

The core rule is simple: **small experiment -> immediate A/B -> keep only measured wins**. No training idea gets a long GPU run before it has survived a bounded pilot.

## What the evidence already says

1. The historical 2.073B anchor lacks usable historical AdamW moments. Exact old-optimizer continuation is impossible; new Phoenix lineages use fresh optimizer state and explicit lineage metadata.
2. Full-size 350M forward/backward/AdamW/checkpoint mechanics work on the owner's RTX 4070.
3. The precision-correct productive contract is FP32 master parameters + FP32 AdamW moments with BF16 autocast compute.
4. The measured FP32-master full-model compute-loop winner so far is seq256 / micro4 / accum1 / no gradient checkpointing / torch.compile(default), about 7.45k tok/s compute-loop on the owner's RTX 4070. This is not an end-to-end training rate.
5. Early chat SFT can improve conversational metrics, but the existing development evidence still exposes a foundation/knowledge weakness. Therefore chat tuning alone is not enough.
6. A 10B-token total is a ceiling, not a release target.

## The fast route

### Lane A — Foundation / knowledge

Use the immutable 2.073B anchor as source. Run bounded full-model refreshes on admitted, pinned, deduplicated high-quality foundation data.

Current approved baseline families:
- FineWeb-Edu EN
- FineWeb EN
- FineWeb-2 DE

Do not add FineMath or another source until the executable source policy admits the exact revision.

Start with small quality gates. Do not jump directly to hundreds of millions or billions of new tokens. At every gate compare:
- held validation loss;
- public development free-answer correctness;
- DE/EN knowledge-choice;
- NIS2 development categories;
- repetition/EOS/generation health.

If a gate is neutral twice in succession, change data/recipe before spending a much larger token budget.

### Lane B — Targeted knowledge repair

Foundation pretraining is broad but slow. After the first foundation candidate shows stable language behavior, run a small **reviewed human knowledge-SFT** lane on material that is not copied from evaluation prompts.

Required coverage:
- everyday factual questions;
- elementary and intermediate arithmetic;
- science/physics/astronomy;
- aviation/transport;
- short explanations;
- German and English.

Use response-only loss with explicit EOS. Start by training only the upper 3–4 blocks at low LR. Mix a small amount of verified foundation replay so the model does not trade language capability for memorized answer style.

A knowledge-SFT candidate is useful only if raw free answers improve. Lower SFT validation loss alone does not qualify it.

### Lane C — Chat behavior

After Lane A/B produces the best capability checkpoint, run the cheap human-only OASST2/OASST1 chat recipe on top of that checkpoint.

Train:
- instruction following;
- multi-turn state;
- concise/long answer control;
- uncertainty;
- role boundaries;
- EOS/stop behavior;
- low repetition;
- bilingual DE/EN dialogue.

Use small hyperparameter sweeps, not one long run. Reuse the existing Phoenix public-development A/B and replay-protected SFT infrastructure.

### Lane D — Serving quality

Do not make the model learn what runtime code should solve.

Helios runtime should provide:
- one canonical chat template shared by SFT and inference;
- KV cache;
- streaming;
- EOS and role-boundary stop handling;
- bounded context truncation;
- conservative repetition guard;
- deterministic checkpoint/tokenizer manifest;
- local API + browser UI.

These improve product quality and latency without pretending they improve model knowledge.

## Optional product accelerator: local retrieval/tools

For the fastest useful Helios product, keep model-only benchmarks separate but allow the product to use local retrieval and deterministic tools. This can supply current or project-specific facts without importing foreign model weights.

Never report retrieval/tool results as base-model intelligence.

## Promotion vector

A candidate can replace the current development leader only when it improves the primary target and does not materially regress the protected dimensions.

Primary target order:
1. raw free-answer correctness;
2. instruction following / exact-generation;
3. DE/EN conversational quality;
4. EOS/repetition health;
5. held validation / language preservation.

Protected dimensions:
- German;
- English;
- math;
- science/knowledge;
- code;
- multi-turn;
- generation health.

Do not collapse these into one synthetic score for final promotion.

## Compute policy

- Short diagnostics: eager may be faster because compile warm-up dominates.
- Sustained full-model runs: use the fastest precision-correct compiled shape only after parity/resume checks.
- Chat/knowledge SFT: train a small upper-block subset first; use full-model SFT only if measured results justify it.
- Checkpoint every bounded recovery interval and keep verified rolling generations.
- Never mutate optimizer/update geometry inside an existing lineage merely to chase throughput.

## Failure-driven development

When an evaluation fails, create training examples for the **capability class**, not copies or paraphrases of the failed evaluation case.

Examples:
- arithmetic failure -> new range-disjoint arithmetic curriculum;
- astronomy failure -> unrelated reviewed astronomy explanations;
- instruction failure -> structurally similar but content-disjoint instruction tasks;
- repetition/EOS failure -> clean response-only conversations with explicit EOS.

This is the main speed advantage: every GPU run attacks a measured weakness instead of adding generic tokens blindly.

## Immediate sequence

1. Finish the current precision-correct bounded foundation refresh and evaluate it.
2. Keep it only if the development vector improves or provides a justified foundation-loss win with no capability regression.
3. Build the reviewed, contamination-screened knowledge-SFT pack while GPU work is unavailable.
4. Run a tiny upper-block knowledge-SFT sweep with replay protection.
5. A/B raw free answers immediately; kill non-improving candidates.
6. Re-run the already successful cheap chat-SFT recipe on the best capability checkpoint.
7. Qualify the winner in the real Helios multi-turn runtime.
8. Only then decide whether another larger foundation stage is worth the compute.

## Reality check

A 350M model can become a useful compact assistant, but it will not reach frontier-model breadth through clever post-training alone. The fast objective is therefore **a clearly useful, stable, bilingual local assistant**, not fake parity with multi-billion-parameter systems.

The project's advantage is iteration speed: small model, local GPU, strong eval discipline, cheap SFT, and a runtime that adds retrieval/tools without corrupting model-only claims.
