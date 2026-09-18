# Phoenix 350M — progress and evidence, 18 September 2026, v3

**Status: experimental only; PR #118 stays DRAFT and unmerged. No new trained 350M weights, approved training or proven answer-quality improvement.** This ledger distinguishes completed engineering checks from actual model capability.

## Foundation and model quality

- Original SHA-pinned immutable checkpoint: 2,073,161,216 previously processed tokens, 104,832 recorded steps, original archive SHA256 `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`. This is ~20.7% of the **planned** 10-billion-token pretraining budget, not 20.7% of model quality or finished work.
- [Actual read-only optimizer audit](https://github.com/Cola968/neral-helios/actions/runs/35345090808): archive has BF16 model tensors, but no usable native AdamW moment entries. It **cannot exact-resume**; missing moments cannot be reconstructed. No unsupported inference about historical moment dtype or quality causation.
- [Real 24-case own-weight development probe](https://github.com/Cola968/neral-helios/actions/runs/35343702334): forced-choice correct digit first-ranked for 4/24 instruction-chat cases, 4/24 minimal-chat and 1/24 QA; 0/8 on a separate bare-equation arithmetic subset. This is public development forced-choice only, NOT real-answer accuracy or independent NIS2-v2. Previously observed 2/3 on THREE equations must not be generalized.

## Repaired native source, tested boundaries

1. `train350.py` and `train350_compiled.py` reject SHA/untrusted/model-only/optimizer-incomplete/dataset-or-sampler-mismatched exact resumes **before CUDA checks, output creation, or model allocation**. Original bad checkpoint rejected by [real CPU negative test](https://github.com/Cola968/neral-helios/actions/runs/35347494772); native main gates were mocked only to inject the real verified defect, not to simulate successful full training.
2. `window_sampler350.py` chooses a uniform ticket across ALL valid token windows rather than oversampling small shards. Both trainers agree on actual Torch CPU batch contents, final window and partial-token masking in [real CPU synthetic-data test](https://github.com/Cola968/neral-helios/actions/runs/35347661572). Historical bias magnitude is **unknown** until the original shard sizes are audited. Validation identity includes the changed sampler version; historical losses cannot be compared across versions without qualification.
3. The native exact-resume preflight additionally checks the serialized AdamW `betas`, `eps`, `weight_decay`, fused/maximize/AMSGrad flags and a finite bounded previous LR against the declared run contract. This fixes an actual override hazard: `optimizer.load_state_dict()` restores group options, not just moments. No claim that a full 350M optimizer can fit in RTX 4070 memory or be trained at target TPS.
4. [Actual PyTorch CPU positive test](https://github.com/Cola968/neral-helios/actions/runs/35348554766) used a **random synthetic 82,304-parameter native-layout model**, one genuine fused AdamW step, a SHA-pinned `resume.pt` and a further matched step. All model tensors and optimizer slots matched an uninterrupted CPU run bit-for-bit; RNG sampler replay matched. Five tampered AdamW options were rejected. This proves a small genuine serialization/restore path, **not** a successful real 350M GPU resume, loss gain or chat improvement.
5. `checkpoint_journal.py` avoids falsely labeling model-only new states as resumable. Initial archival SHA receipt explicitly says structural verification was not performed. No historical checkpoint was overwritten.

## CI evidence and limits

- [Final complete synthetic regression on the user's PC](https://github.com/Cola968/neral-helios/actions/runs/35349306482), exact source commit `4012d87786351578fb9d9fd292c54df87c521d72`: **188 Phoenix tests (2 skipped), 40 SFT, 7 storage and 10 journal**, 245 total, 243 passed / 2 skipped, no test failures. Python syntax compilation succeeded. Entire job used WSL CPU only (`CUDA_VISIBLE_DEVICES=-1`), no source data or real model training. The two skips are not test passes.
- A prior hosted run at `8754c34` [failed](https://github.com/Cola968/neral-helios/actions/runs/35348558938) because earlier fake-checkpoint fixtures omitted newly required AdamW group fields. The fixture was corrected in commit `4012d877`. Subsequent hosted runs at `4012d877` and `5fb4c885` report failure with **no available job steps** and blob-log retrieval errors; exact hosted infrastructure cause is not established. Do not call hosted PR checks green. The successful *self-hosted* full suite above is the directly observable regression evidence.
- One earlier self-hosted full-regression workflow invocation failed before tests due a PowerShell/WSL command-variable quoting error; the workflow was corrected and rerun successfully. Keep failures in the record instead of cherry-picking.

## Stop/go and next engineering/research gates

- **Exact resume of this real 2.073B checkpoint: CLOSED**, regardless of the small-model positive test. Seek a genuinely complete native checkpoint with pinned bytes or design separately authorized, isolated weight-only warm start with new optimizer/lineage/LR/output and baseline evaluation. Do not forge optimizer history or relabel old tokens.
- **SFT: CLOSED** pending actual independently reviewed source provenance/licensing, privacy/content decisions and a newly independently authored frozen bilingual holdout with solution rubrics and overlap checks. Existing public development cases cannot become the holdout.
- **Training/release: CLOSED** pending explicit run approval, 3-snapshot stable GPU preflight, storage guards, bounded pilot, real measured throughput and blind original/candidate **free-generation** A/B quality on fixed cases. The direct native trainer paths alone do not implement the GPU-idle or journal disk guard.
- Direct path preflight still checks RNG-key presence rather than full RNG-payload validity; full 350M GPU resume remains untested. Keep [issue #119](https://github.com/Cola968/neral-helios/issues/119) open for real full-checkpoint evidence, remaining guard review and operational approval.

**Bottom line:** the source and reproducibility improved in verified, specific ways; intelligence, quality and training speed have not yet been shown to improve. Original weights and private data remain untouched.
