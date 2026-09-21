# Phoenix weight-only pilot — operator contract (updated 21 September 2026)

**Experimental implementation; training is NOT authorized or started.** Separate create-only weight-only pilot, NOT the historic AdamW trajectory. Never claim source review or independent holdout complete without actual human verification. GitHub Actions only run CPU tests/read-only inspections; they do not launch the pilot.

## New precision correction (not a quality result)

The original proposed pilot placed the 350M trainable parameters in BF16 and constructed AdamW from those BF16 parameters. At LR 2e-5, many small updates risk rounding away; this is a numerical risk, **not proof that BF16 caused the historical model's poor answers**. The experimental entrypoint now loads the historical SHA-pinned BF16 source *into FP32 master parameters*, creates a NEW fused AdamW optimizer with FP32 moments, and uses CUDA BF16 **autocast only for forward/evaluation compute**. The source's historically lost precision cannot be recovered. Runtime checks reject BF16 master parameters, incomplete/non-FP32 moments and checkpoints saved in incompatible dtypes. VRAM reserved above 10 GiB stops the pilot; actual RTX 4070 fit and throughput remain untested. Checkpoint and optimizer storage requirements increase; no budget, thermal or feasibility claim follows from the change.

**All old sample manifests and approval hashes are obsolete.** The exact `run_contract` must now include `master_parameter_dtype='float32'`, `adamw_moment_dtype='float32'`, and `compute_autocast_dtype='bfloat16'`. Do not silently edit a prior manifest or reuse its approval digest. No real manifest/approval exists at present.

## Immutable source and data checks

`phoenix_weight_only_pilot.py` verifies SHA-pinned original 350M source (`f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`), tokenizer (`4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`), uint16 train/validation shard identities and identical-byte disjointness, dimensions, approval, source-review and frozen-holdout checksums, output nonexistence and disk reserve before GPU access. It checks historical architecture, BF16 source tensor layout and tied embeddings on CPU. The original 2,073,161,216 processed tokens remain provenance only; new run starts at zero steps/tokens. No old moments are recovered or faked. Source archive remains read-only, output is separate.

Three stable idle GPU snapshots, RTX 4070/BF16/free-VRAM preflight, runtime thermal checks, new-run exact SHA-pinned optimizer/RNG/sampler/contract preflight and SHA journal are enforced. None guarantees real-model VRAM fit, quality, source immutability after preflight or crash recovery under power loss. An OOM can leave a new incomplete directory; never redirect into the original assets.

## Exact operator manifest (schema 1)

Only after substantive independent source-provenance, license, privacy and content review **and** an independently authored, frozen bilingual free-answer holdout with answer rubrics and leakage checks may a human operator create local source-review, holdout and approval files. The manifest contains exactly:

- `schema` integer `1`; `approved_for_bounded_pilot`, `human_source_review_complete`, `human_holdout_review_complete` true only after genuine review.
- `source_sha256`, `tokenizer_sha256`: locked digests above. `review_sha256`, `holdout_sha256`: different SHA-256 digests of actual local evidence files.
- `train_identity`, `validation_identity`: exact `corpus_identity` values, train seed `seed+1`, validation seed `seed+999`, validation microbatch and `eval_batches`; identical shard byte contents disallowed. Hashes do not exclude semantic contamination.
- `output`: canonical absolute, initially nonexistent separate output; `target_new_tokens`, `stop_after_new_tokens`: exact proposed total NEW token target and invocation stop.
- `run_contract`: exact `run_contract` with seq/microbatch/accumulation/seed/LR/cosine schedule/weight decay/warmup/gradient checkpointing, plus `initialization='weight_only_new_optimizer_v1'`, `source_sha256`, approved `compile_mode` and **the three new precision fields stated above**.

CLI requires `--execute`, `--trusted-local-checkpoint`, explicit source/assets/train/valid/output/approval/source-review/frozen-holdout paths, independently obtained exact manifest `--approval-sha256`, and `--target-new-tokens`. Private data, weights and answers must stay off GitHub. A hash and boolean field are not evidence of real human approval.

## Run bounds, recovery, open gates

Maximum 262,144 NEW tokens, seq <=512, microbatch <=2, grad accumulation <=16, <=4,096 tokens/update, peak LR in (0,1e-4]. Default seq128/microbatch1/accum4/LR2e-5 is a configuration, not an optimal setting. Intermediate stops must align to complete updates; the terminal remainder may be partial. No historical counter may be added to new counters. Existing destination is only accepted for its own exact pinned complete FP32 `resume.pt` with matching origin, run contract and approval; original checkpoint never qualifies. Current SHA/lineage bind the current invocation stop: crash recovery to that SAME approved stop is contemplated, deliberate pause/extend is not supported.

Before real training: complete genuine data review and independent frozen free-answer evaluation; obtain explicit hash-bound approval of the **new precision contract**; establish separate GPU/VRAM/disk feasibility, baseline correctness and candidate correctness on identical frozen cases. Do not infer intelligence from a lower validation loss, forced-choice ranks or cleaner output. Keep PR #118 DRAFT and Issue #119 OPEN.

Prior synthetic-only regression (before precision correction): [#35353441112](https://github.com/Cola968/neral-helios/actions/runs/35353441112), 268 tests, 266 passed and two skipped, plus tiny random-model AdamW replay. This does **not** test FP32/BF16 full-model feasibility or quality. New precision tests and any new CI results must be reported separately once verified.
