# NERAL Phoenix 350M — verified progress v4 (18 September 2026)

**Draft PR #118 only. No new Neral weights, SFT, GPU optimizer updates, quality gain, measured throughput improvement or training authorization. No source model weights, private training texts or private answers uploaded to GitHub. Original checkpoint immutable.** This report supersedes v3 for current engineering status; v3 remains as an earlier evidence snapshot.

## Baseline remains unchanged

The original SHA-256 pinned own checkpoint `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7` records **2,073,161,216 historical processed tokens**, step 104,832, BF16 model-state tensors and missing/empty native AdamW moment state. An optimizer dictionary key alone does not make it exactly resumable. An earlier 24-item public forced-choice developmental probe ranked the correct digit first for 4/24 instruction-chat cases, 4/24 minimal-chat, 1/24 Q/A and 0/8 distinct bare equations. Those probes are not independent evaluation, generated-answer correctness or model quality improvement. See `QUALITY_FINDINGS_2026_09_18_V2.md` and `OPTIMIZER_RESUME_AUDIT_2026_09_18.md`.

## Source bug found and fixed in this work period

`train_full_with_journal.py` invoked `verify_weight_moment_mapping(source)` after the native guard introduced mandatory keyword-only `expected_run`. A valid optimizer would cause `TypeError` before reaching training: the native direct paths were guarded but the journal wrapper was broken. The wrapper now calls a separately testable `validate_journal_source` which checks native schema and sampler provenance, runs full AdamW structural checks and calls the mapping verifier with **`expected_run=source['run_contract']`**. The direct native trainer still independently verifies that saved contract against CLI parameters and actual data hashes. Five synthetic regression tests check the real wrapper dispatch signature, both validators' order and refusal propagation. This fixes a genuine wrapper regression, not real-model quality.

## Exact-update boundary invariant

Native trainers compute an abbreviated final optimizer update when `run_until` falls inside a normal update. Such an *intermediate* partial update changes optimizer-step count and gradient accumulation compared with an uninterrupted run. `native_resume_preflight350.require_exact_update_boundary` now enforces, before CUDA allocation on both native **resume** entrypoints:

- A nonterminal checkpoint must satisfy `processed_tokens == step * (seq * microbatch * grad_accum)` using the pinned run contract.
- If stopping before the total target, the next `run_until` must be an integer multiple of that update size.
- A final `run_until == target_tokens` may have a remainder because there is no later resume.

This is **not** the same as preventing every fresh native launch from requesting an intermediate partial stop: direct fresh-trainer CLI still permits creating such a checkpoint, but the next exact resume now explicitly rejects it. An early fresh-launch check is still recommended. Synthetic tests cover partial past states, miscounted steps, off-boundary future stops, final remainder, invalid dimensions and a normal aligned state. The original 2.073B checkpoint remains rejected for missing moments before any alignment claim.

[Complete post-fix self-hosted PC CPU regression #35350728089](https://github.com/Cola968/neral-helios/actions/runs/35350728089): **196 Phoenix + 40 SFT + 7 storage + 10 journal = 253** synthetic/code tests, **251 passed, 2 skipped**, no failures. The same run replayed one real tiny-model PyTorch fused AdamW checkpoint on CPU and reproduced the next model/optimizer state bit-for-bit. This is not 350M-GPU reproducibility.

## Actual historical weight-only warm-start readiness (no launch)

`phoenix_warmstart_readiness.py` is a new separate, opt-in, CPU-only read-only preparation **probe**, not a training command. It SHA-checks the exact trusted original checkpoint before deserializing, verifies locked architecture, tokenizer, schema, BF16 tensor shapes/names against a meta-device model layout, validates the tied embedding/head tensors contain identical values, confirms native AdamW moments are empty, and requires a separate nonexistent proposed output outside the original checkpoint folder. It creates neither output directory nor model nor optimizer weights, and its report says training is unapproved.

[Original-weight on-PC probe #35351114627](https://github.com/Cola968/neral-helios/actions/runs/35351114627) **PASSED**. Six synthetic warm-start contract tests passed. The actual original checksum, locked weight layout with **267 model-state tensor entries** (not 267 model parameters), tied head equality and separate nonexistent output were verified. It recorded **2,073,161,216 historical tokens**, but proposed new-run counters **0 new tokens and 0 optimizer steps**. No private text, model tensors or answers in GitHub logs; no original weights modified, GPU, training or output folder. The legacy trusted pickle loader was used only after full SHA and explicit trust.

**Important:** readiness does not constitute an implemented runnable 350M warm-start trainer. Fresh optimizer construction, LR plan, independent data/license/privacy approval, a frozen bilingual holdout and quality A/B still need separate implementation or approval. This checkpoint is a weight source, never an exact AdamW resume.

## Stop/go, status and evidence limits

- `source_weight_readiness=true` (CPU, exact pinned model only), `exact_original_adam_resume=false`, `warmstart_trainer_implemented=false`.
- `training_approved=false`, `new_training_tokens=0`, `new_model_weights=false`, `quality_gain_verified=false`, `60k_to_70k_tps_verified=false`, `production_modified=false`.
- Full 350M native GPU start/resume, RAM/VRAM/thermal budget and stable FP32 optimizer throughput have not been demonstrated. No new independently authored frozen bilingual holdout, substantive human data/provenance/privacy/license/content review, semantic decontamination or approved pilot exists.
- Prior GitHub-hosted PR checks can report infrastructure failure without steps or retrievable logs; only the cited self-hosted runs' successful assertions are claimed as tested. Do not merge PR #118 or close issue #119 on this evidence alone.

Next engineering target: explicitly separated create-only warm-start trainer with a **fresh** optimizer, new lineage, pinned independent dataset, 3-snapshot GPU-idle and storage guards, and a small opt-in numerical pilot *only after data/run authorization*. Do not invent missing optimizer moments, relabel historical token counters or promote without blinded real free-answer improvement on a frozen holdout.
