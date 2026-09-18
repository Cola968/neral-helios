# NERAL Phoenix: native trainer repair — 18 September 2026

**Experimental branch only; PR #118 remains DRAFT. No new weights, no training, no measured answer-quality or throughput improvement.**

## 1. Actual defect: unusable optimizer state in 2.073B snapshot

The immutable own checkpoint at 2,073,161,216 processed tokens (SHA256 `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`) contains model BF16 tensors and a top-level optimizer dictionary, but its native AdamW `state` has no usable moments. [Original CPU audit](https://github.com/Cola968/neral-helios/actions/runs/35345090808) and [real negative contract test](https://github.com/Cola968/neral-helios/actions/runs/35345880926). A model-only snapshot cannot exactly continue a 104,832-step optimizer trajectory; discarded moments cannot be reconstructed.

Both `frontier350/train350.py` and `frontier350/train350_compiled.py` now invoke the same `native_resume_preflight350.read_verified_resume` **before CUDA checks, model construction, optimizer allocation, output directory creation, or training**. It requires explicit trust and full checkpoint SHA; original non-symlinked `resume.pt` in its matching output directory; native config/schema, tokenizer and cumulative LR-target identity; full optimizer and AdamW moments, parameter shapes/dtypes and per-slot optimizer steps; sampler/RNG presence; identical train/validation shard byte digests, sampling version, seed, window length and run hyperparameters. Historical checkpoints without these records are deliberately **not exact-resumable**. Fresh training writes the new metadata in its complete checkpoint; no old optimizer state is manufactured.

`train_full_with_journal.py` forwards its trusted SHA arguments to the now-mandatory native verification instead of accidentally bypassing them. It also rejects legacy/unknown sampler provenance before making a journal receipt. The original is not overwritten in this work.

[Real checkpoint native `main()` negative test](https://github.com/Cola968/neral-helios/actions/runs/35347494772) passed on the self-hosted Windows PC under WSL CPU only. It SHA-checks and loads the actual immutable checkpoint, substitutes its **real failed optimizer-state verification** at the native gate, calls BOTH genuine `main()` functions with temporary synthetic shards, and raises if CUDA availability is consulted first. Both refused with `missing_native_AdamW_moment_state`. The gate is intentionally mocked to inject the verified real state: this is **not** an end-to-end valid-resume, GPU or training test.

## 2. Actual code-level data distribution defect: shard-uniform selection

Former code selected every `.bin` file with equal probability, then selected a start uniformly *inside* that file. When a full shard contains 100 valid starts and a short residual shard contains one, old selection gives the residual shard **50% of draws**, rather than its correct **1/101** share. Actual historic overrepresentation magnitude has not been measured because historic shard lengths for the full training stream have not been audited here.

`window_sampler350.WindowSampler` uses one seeded `randrange(sum(len(shard)-seq))` ticket mapped to a shard and an exact local start. Both native trainers use this shared logic: the compiled trainer caches the map for a given sequence length; the legacy trainer reuses a per-run training map and per-validation map. Sampling identity is stored in fresh checkpoints and included in validation identity. Old validation losses and old training RNG states must not be treated as directly comparable across this change.

[Real Torch CPU sampler integration test](https://github.com/Cola968/neral-helios/actions/runs/35347661572) passed: two uneven synthetic uint16 shards with **1 and 100 valid windows** yielded exact parity of X/Y batches in the legacy and compiled paths, correct autoregressive shift, all 101 starts including the last one, and correct partial-update target masking. `pin_memory` was mocked only for CPU compatibility; it is a CUDA transfer optimization and does not alter the selected samples. This is a functional test, **not measured training throughput or model quality**.

The legacy training loop was also corrected to compute the final optimizer update from actual remaining supervised tokens, mask excess targets to `-100`, weight microbatch losses by supervised count and increment the processed-token counter *after* the optimizer step. Nonfinite gradient clipping raises before an update. No GPU training verified these changes; synthetic tests and real CPU functional checks are the current evidence boundary.

## 3. Outstanding stop/go

- Source authorization and human review for SFT remain pending, and a separately authored, frozen bilingual holdout is missing. Do not start SFT or a multibillion-token continuation.
- No positive end-to-end native checkpoint resume has been executed, no GPU/VRAM/thermals pilot on these rewritten trainers, no throughput benchmark and no evidence of improved free answers. Direct entrypoints do not replace the three-sample GPU-idle preflight or journal/disk guard required for an approved run.
- Original 2.073B snapshot is useful for **read-only inference or a separately authorized, explicitly marked warm-start experiment**, not as a native exact AdamW resume. Such an experiment needs a distinct output, separately approved LR, disk/GPU guard and baseline/candidate answer-quality comparison; no warm-start path is implemented or approved here.
- Additional native launch review: validate a fully populated genuine native AdamW checkpoint layout, ensure RNG-state payload validity, exercise an opt-in isolated positive short resume, and audit actual historic shard sizes/validation distribution before concluding the full pipeline is production-ready.

**Status:** direct model-only resume is rejected, sampler/partial-update code changed and CPU-tested, old weights untouched, quality improvement unverified. Keep PR #118 draft and issue #119 open for positive resume and approval gates.
