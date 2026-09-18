# Phoenix weight-only pilot — operator contract (18 September 2026)

**Experimental implementation; training is NOT authorized or started.** This is a separate, create-only pilot, not a continuation of the historic AdamW optimization trajectory. Do not create a manifest falsely stating that source review or the independent holdout is complete. GitHub Actions in this branch only execute synthetic CPU tests and read-only inspections; no Action launches this trainer.

## What changed

`phoenix_weight_only_pilot.py` is an explicit executable entry point, distinct from native exact `--resume` or SFT. It verifies the original SHA-pinned 350M source (`f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`), tokenizer SHA (`4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`), uint16 train/validation shard hashes and disjoint identical shard bytes, run dimensions, approval manifest, source-review and frozen-holdout checksums, output nonexistence and disk reserve *before GPU access*. It checks full original model architecture, BF16 tensor layout and tied embeddings on CPU. The 2,073,161,216 historical tokens are **source provenance only**: newly processed tokens and new optimizer steps start at zero. A new fused AdamW optimizer is created with no inherited moments. Source checkpoint is read-only and its bytes are rehashed in the final summary.

The pilot enforces three idle GPU measurements, RTX 4070/BF16/free VRAM checks, training-time temperature checks, a separate destination, complete optimizer/RNG/sampler/data/contract checkpoint preflight on any new-pilot resume, and a SHA-checked checkpoint journal with no automatic file removal. These safeguards do NOT prove model fit in VRAM, stable training throughput or ability. An out-of-memory failure may leave an empty new output; it never authorizes reusing the original source directory.

## Exact manifest fields (schema 1)

The human operator must independently create **local** review and frozen-holdout files, finish actual provenance, licensing, privacy, quality and non-overlap reviews, then create a manifest containing *exactly*:

- `schema`: integer `1`; `approved_for_bounded_pilot`, `human_source_review_complete`, `human_holdout_review_complete`: Boolean `true` **only after verified human decisions**.
- `source_sha256`, `tokenizer_sha256`: the locked full 64-character digests above.
- `review_sha256`, `holdout_sha256`: SHA-256 of the actual local review and independent holdout files. The two must differ.
- `train_identity` and `validation_identity`: exact `corpus_identity` outputs for proposed shard sets and sampler version. Train seed `seed+1`, validation seed `seed+999`, validation batch = microbatch and batches = eval-batches. No identical train/validation shard bytes. These SHA checks do not detect paraphrase/semantic contamination.
- `output`: canonical absolute and *not yet existing* new destination path (for fresh start); `target_new_tokens`, `stop_after_new_tokens`: exact proposed new-token cumulative target and current invocation end.
- `run_contract`: exact `run_contract` output (`seq`, `microbatch`, `grad_accum`, `seed`, `lr`, `schedule='cosine'`, `weight_decay`, `warmup_frac`, `gradient_checkpointing=True`), plus `initialization='weight_only_new_optimizer_v1'`, `source_sha256` and `compile_mode` (`none` or `default`). A changed parameter or compiler mode invalidates the authorization.

The command line also requires `--execute`, `--trusted-local-checkpoint`, explicit source/assets/train/valid/output/approval/source-review/frozen-holdout paths, `--approval-sha256` (full digest obtained **independently from the exact manifest bytes**), and `--target-new-tokens`; optional `--stop-after-new-tokens` selects the approved end. No real manifest is included in the repository; private text/answers, weights and data should not be uploaded. A SHA hash is not an identity signature; claims of human review must be substantiated by the actual contents and provenance of local evidence, not generated flags.

## Upper limits and state rules

- Maximum 262,144 new tokens, seq <=512, microbatch <=2, accumulation <=16 and <=4,096 tokens per update; peak LR must be >0 and <=1e-4. Default seq128/microbatch1/accumulation4/LR2e-5 is a *configuration*, not a demonstrated optimal training setting.
- An **intermediate** `stop_after_new_tokens` must be a multiple of tokens/update; a terminal final-target remainder is permitted. Full old training counters must never be added to the new run counters or passed off as new training.
- New destination is create-only. Existing output is only accepted with its own explicit SHA-pinned `resume.pt`, *complete* AdamW/RNG/data contract and `warmstart_lineage`. Original model-only archive cannot be passed as `--resume`.
- **Current limitation:** manifest and lineage bind the exact current invocation stop and approval SHA. Crash recovery to the *same approved stop* is supported; extending a deliberately staged early stop to a later stop is not yet supported without redesigning the authorization/lineage protocol. Do not claim arbitrary pause/extend/resume is proven.
- Model and approved source data could change on disk between preflight and use; checksum preflight and separate output are not a filesystem snapshot or independent semantic data audit. CUDA execution, 350M GPU memory feasibility, checkpoint power-loss recovery of this entrypoint, answer-quality improvement and any purported 60–70k training tokens/s have **not** been measured.

## Verified evidence and release gate

[Synthetic CPU contracts #35352730295](https://github.com/Cola968/neral-helios/actions/runs/35352730295): 266 total tests, 264 passed, two skipped. [Final hardened CPU source and genuine tiny PyTorch lineage/journal #35353441112](https://github.com/Cola968/neral-helios/actions/runs/35353441112): **268 tests, 266 passed and two skipped** plus 82,304-parameter CPU model checkpoint/optimizer replay and SHA journal. [Real existing 350M read-only readiness #35351114627](https://github.com/Cola968/neral-helios/actions/runs/35351114627) verified its actual weight layout but **did not test this pilot end-to-end**.

Do not launch a real GPU pilot until an independently authored and frozen bilingual free-answer holdout, substantive human data review, SHA-bound operator run approval, independently checked CUDA/disk budget and a predeclared bounded baseline/candidate free-generation A/B are complete. Never use validation loss or synthetic test success alone as evidence that Neral became more intelligent. Keep PR #118 in DRAFT and issue #119 open.
