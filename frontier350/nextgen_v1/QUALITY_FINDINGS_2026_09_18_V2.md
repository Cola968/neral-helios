# Neral 350M: real-checkpoint diagnosis v2 — 18 September 2026

**Status: NO verified model-quality improvement, NO SFT, NO authorization for long training.** This is public development probing, not an independently authored or frozen test set. The previous 3-question positive observation does **not** generalize to the subsequent 24 questions.

## Model / execution identity

All three investigations used exactly the same original own Neral 350M checkpoint, with `processed_tokens=2,073,161,216` and SHA-256 `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`, plus the pinned own BPE tokenizer. The Windows self-hosted runner used WSL with `CUDA_VISIBLE_DEVICES=-1`: CPU only, no optimizer, GPU training, weight mutation, network data transfer or GitHub artifact upload. Full generated answers and per-item likelihoods remain in `D:\NeralPrivateAssets\phoenix\`.

1. [Original raw/guarded + KV parity](https://github.com/Cola968/neral-helios/actions/runs/35342497944) passed execution: **raw 0/3 strict exact, guarded 0/3 strict exact** at 16 output tokens, two KV parity points numerically and top-1 passed. Strict digit equality is a formatting-sensitive measure, not a human semantic adjudication.
2. [Original teacher-forced digit-choice ranks](https://github.com/Cola968/neral-helios/actions/runs/35342740102) passed: right digit ranked first **1/3 with existing chat context and 2/3 as bare equation**. This never showed reliable math or actual free-answer correctness.
3. [Five-format replay / ablation](https://github.com/Cola968/neral-helios/actions/runs/35343435062) passed, with exact reproduction of the prior raw output. For the same three cases, free greedy answers began with the correct digit **0/3 with current chat, 0/3 with minimal chat, 2/3 as an equation**, but no format delivered a strictly exact one-digit answer. These are limited, non-independent development samples. Each format also changes prompt wording and token context, confounding causal attribution.
4. [24-item domain evaluation](https://github.com/Cola968/neral-helios/actions/runs/35343702334) passed using fresh, publicly committed questions: eight arithmetic, eight everyday, four astronomy, four aviation; twelve German and twelve English. Each choice was scored from the same verified checkpoint, among ten digit candidates with identical candidate token counts within each prompt. **Forced-choice only; no free-generation and no independent holdout.**

## Larger development-suite results

Correct digit ranked first among 0–9 (counts / denominators):

| Domain | Cases | Instruction chat | Minimal chat | Question/Answer | Bare equation |
|---|---:|---:|---:|---:|---:|
| Arithmetic | 8 | 1/8 | 2/8 | 0/8 | **0/8** |
| Everyday | 8 | 1/8 | 0/8 | 0/8 | not tested |
| Astronomy | 4 | 1/4 | 1/4 | 0/4 | not tested |
| Aviation | 4 | 1/4 | 1/4 | 1/4 | not tested |
| **Total** | **24** | **4/24** | **4/24** | **1/24** | 0/8, arithmetic only |

These counts are **not chatbot correctness**, and none demonstrate quality improvement. The 2/3 equation observation from the original tiny sample is contradicted as evidence of reliable equation skill by the subsequent **0/8 new arithmetic examples**. Do not advertise equation prompting, the guard decoder, Wikipedia retrieval, or KV caching as a model-knowledge fix. Arithmetic difficulty, prompt distribution and candidate framing are all relevant confounders; none establishes a unique root cause.

## Source-level facts and remaining cause hypotheses

- `build_frontier_bpe_data.py` serializes foundation documents with BOS/EOS; it does **not** train a systematic `User: ...\nAssistant:` chat distribution. `helios350_preview.py` and guarded preview nonetheless use that format for chat; supervised training via `sft_contract.py` was planned but **no pilot run completed**.
- The planned pretraining schedule is 10B tokens; this exact checkpoint represents 2.073B processed tokens, ~20.7% of that target. A target is not achieved training. This and distribution mismatch are hypotheses, not sufficient causal attribution for poor performance.
- `helios_quality_diagnostic.py` confirms a limited native/KV parity check and exact raw replay; there is no evidence here of a checkpoint-loading or cache divergence causing the poor results. This does not rule out every implementation bug.
- The 24 new public questions were committed before evaluation and were not used to update weights. Nevertheless they are transparent development fixtures and MUST NOT become an independent holdout or be recycled into training/evaluation claims.

## Precise stop/go decisions

**Remain CLOSED:** no unbounded 2–3B token continuation, no synthetic pass/fail labels presented as reviewer decisions, no SFT from the 1,200-row candidate without human provenance/privacy/content approval, no replacement of a missing frozen holdout with published development questions, no alteration of the known original checkpoint.

**Required next experiment:** Independently author and freeze 24–100 private bilingual multi-domain cases and rubrics *before* candidate modifications; conduct train/validation/holdout exact and semantic overlap screening. Human-review the candidate source and licenses, starting with the 24-case offline sample and representative full-source auditing. Once explicit hash-bound approval and stable three-sample GPU preflight exist, run a genuinely bounded, isolated answer-only SFT experiment; assess raw own-weight free-answer correctness and stability on the same frozen evaluation set versus the immutable baseline. Validation loss alone, a forced-choice diagnostic, or cleaner formatting cannot qualify as a quality gain.

**Current confirmed outcome:** better diagnosis, unchanged model weights, unsatisfactory model performance, no independently established improvement.
