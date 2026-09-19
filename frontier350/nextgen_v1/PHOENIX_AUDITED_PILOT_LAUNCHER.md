# Phoenix audited pilot launcher — experimental; no training approval

`phoenix_audited_pilot_launcher.py` is an **optional, separate entrypoint** around the existing `phoenix_weight_only_pilot.py`. It does not run automatically, is not referenced by a training workflow, creates no human approval and does not alter old Neral 350M weights or training shards. **The direct pilot entrypoint remains available and does not run these additional mandatory-for-the-launcher checks.** Use the audited launcher, not the direct entrypoint, for the full protection described here.

## Prerequisites and intended invocation

Follow the original `WEIGHT_ONLY_PILOT_OPERATOR_CONTRACT.md`: genuinely review dataset provenance/licensing/privacy and semantic contamination, create and freeze an independently authored bilingual DE/EN evaluation holdout with the exact `phoenix_blind_ab_audit.validate_holdout` JSON schema, independently SHA-pin the unchanged original checkpoint and all proposed data, explicitly approve the exact bounded training manifest and verify local disk/GPU/temperature capacity. The operator approvals and actual independent holdout were **not available as of 19 September 2026**. There has been no authorized real invocation.

Only after all of those prerequisites actually exist, supply the original pilot's **entire approved CLI argument set** and change only the script name:

```text
python frontier350/nextgen_v1/phoenix_audited_pilot_launcher.py [THE EXISTING COMPLETE APPROVED PILOT ARGUMENTS]
```

No bypass flag exists in this wrapper. The original pilot parses the arguments and `prepare()` checks explicit `--execute`, `--trusted-local-checkpoint`, own original SHA, tokenizer, reviewed source and holdout files, manifest SHA and declared human decisions, fresh destination, bounded training contract, train/validation identities and disk **before** either added scan or GPU use.

## What now blocks training, in order

1. The launcher reads **all bytes** of the train and validation uint16 `.bin` files, rejecting illegal token IDs (outside `0..16383`), odd lengths, missing causal windows, byte-identical shards within/across splits and observed same-read file mutation. It rebuilds the approved native corpus identities from the scanned filenames, content SHA, token counts and run/sampler parameters. An identity mismatch aborts before model creation.
2. **Mandatory frozen-holdout contamination gate:** `phoenix_holdout_gate.require_zero_exact_overlap` opens the actual approved `--frozen-holdout`, verifies the exact holdout SHA against the original pilot's plan lineage, validates the bilingual JSON schema, loads the SHA-pinned own tokenizer and verifies its 16,384-token vocabulary. It then scans every training and validation shard for both raw-question and full chat-template token sequences using `phoenix_holdout_overlap_audit.audit`. Any hit, wrong SHA, invalid token ID, malformed data or unexpected diagnostic status **stops the launcher without calling the training routine**. No matching question, case ID, shard path or answer is included in the exception message or standard success output.
3. Only when both stages succeed does the wrapper call the original pilot `main(argv)` with unmodified args; that function repeats its full original approval and data-identity preflight before Torch/CUDA. No special permission is inferred from a zero-overlap result.

The overlap scan is an additional complete corpus pass, so the wrapper can be slow on a large proposed dataset. It is deliberately fail-closed rather than skipping a scan to meet a speed target. All holdout text, shard content, case IDs and checkpoint bytes remain private on the operator's machine; the Python unit tests use synthetic cases and files only.

**Limits:** No exact match does **not** establish independent evaluation. BPE boundary changes, paraphrases, partial copied solutions, related source documents and unscanned data may still contaminate the holdout. Separate human semantic, source-rights and privacy reviews remain essential. Hash checks and repeated scans cannot provide an immutable snapshot against later data mutation, nor authenticate a self-declared human approval. The direct pilot still bypasses the optional extra scans. Nothing here establishes that an RTX 4070 can run the 350M pilot, that checkpoint replay works at full size, that any answers improved or that 60–70k accepted training tokens/s is attainable.

## Verification

The prior launcher passed six separate synthetic tests. This revision adds six synthetic holdout-gate tests, an additional mock launcher order/block test module (three tests), five cross-module toy-shard integration cases using the real gate/holdout-validator/token matcher/shard-audit algorithms but a fake external tokenizer, and extends the original launcher tests to require the gate. The **14 new tests** (six + three + five) passed locally with reconstructed canonical helper functions and fake external dependencies; Python syntax compilation succeeded. Their GitHub equivalents are committed, but GitHub-hosted CI has repeatedly failed before assigning a runner; do not call hosted CI green or the real 350M experiment validated. No owner's genuine shards, private holdout, weights or training were used.

Keep parent draft PR #118 and draft PR #120 open, original weights unchanged, and issue #119 open until real source approval, bounded hardware/run and independent free-answer evaluation are complete.
