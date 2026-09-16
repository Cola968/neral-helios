# Neral NextGen — exact resume A/B qualification (NOT YET RUN ON REAL MODEL)

The offline `tests/test_v7_exact_resume_ab.py` now performs a **real uninterrupted-versus-interrupted continuation** using `v7_recovery.py` on a small CPU model. It serializes/deserializes the recovery state and compares complete output state after further optimization; it is NOT a 350M or RTX4070 test. `compare_exact_resume.py` performs a deep read-only comparison of two *existing* V7 snapshots (all masters, optimizer tensors and groups, torch/CUDA/NumPy/Python/sampler RNG, semantic contract, input batch chain and source/total token counts). Neither tool is a release gate on its own.

## Before any real GPU A/B run

- Identify the **independently verified SAFE** own-weight stage anchor and separately pin its SHA-256, tokenizer SHA, exact model implementation, corpus manifests/shard SHA, trainable block names and optimizer configuration. The V8.1 continuation candidate around 2.073B processed tokens is NOT the last independently SAFE checkpoint. Do not guess a checkpoint path or use one because its filename looks newest.
- Obtain permission for actual local data and ensure training and frozen holdouts are separated. The isolated research branch must not change the production data, checkpoint or validation stores.
- Explicitly allocate two NEW, empty experiment output directories on the user's machine. Never run the two arms in parallel on one RTX4070, never overwrite an existing output or anchor, and do not automatically launch a long or paid job.
- Use identical seed, training contract, optimizer, sample order and initial weights. A short interrupted arm must capture **after a completed optimizer update**, not during accumulation. Confirm saved progress is strictly less than the stage endpoint.

## Required two-arm evidence

1. **Arm A — uninterrupted:** from a verified copy of the same anchor, run a bounded number of actual GPU optimizer updates. Preserve its V7 compact recovery snapshot at an agreed intermediate point *after* the planned interruption point. Do not call this a quality improvement.
2. **Arm B — interrupted:** from a separate verified copy of the identical anchor, run to an optimizer-step boundary, save an exact V7 recovery, terminate the trainer cleanly, then resume explicitly from that recovery into the same B output with matching semantic contract. Complete exactly the same number of updates as A. The pre-interruption and final sample chains must be checked against A.
3. Keep the two **final** V7 snapshots immutable, at the same `new_tokens`, `updates` and `processed_tokens`. Preserve their separately pinned file SHA-256 outside the training output. Do not calculate the claimed independent digest solely from the evidence you are evaluating.
4. Compare the two snapshots with the command below. A `match` is only artifact-state equality; the run logs, true checkpoint provenance, actual source file identity, validation behavior, GPU nondeterminism, and all frozen model tensors also require separate verification. A mismatch is a hard stop, not a cue to loosen tolerances silently.

Run from the repository root using only trusted local V7 files (PyTorch uses pickle for this format):

```bash
python frontier350/nextgen_v1/compare_exact_resume.py \
  --uninterrupted PATH_TO_TRUSTED_ARM_A_FINAL_V7.pt \
  --resumed PATH_TO_TRUSTED_ARM_B_FINAL_V7.pt \
  --uninterrupted-sha256 INDEPENDENTLY_PINNED_ARM_A_SHA256 \
  --resumed-sha256 INDEPENDENTLY_PINNED_ARM_B_SHA256 \
  --anchor-sha256 VERIFIED_SAFE_ANCHOR_SHA256 \
  --acknowledge-trusted-local-pickle
```

The comparer never writes checkpoints, invokes CUDA, launches training, downloads weights, uploads data or incurs paid GPU cost. Exit `0` = equal serialized states, `1` = valid but different, `2` = rejected/invalid input. It rejects missing/empty/symlink files, identical input paths, changed bytes, wrong anchor, missing RNG/counter fields and inconsistent token totals. Because `torch.load(weights_only=False)` can execute malicious pickle content, **never run on untrusted downloaded checkpoint files** even if their hashes match another untrusted message.

## Further mandatory gates

The production trainer and its exact patch chain must undergo a bounded RTX4070 A/B with the **real** verified checkpoint and corpus, including numeric parity across any execution-profile switch. Verify every frozen weight against the anchor, full optimizer-state and sampler equivalence, checkpoint restart across a process boundary, all safety guards, and unchanged evaluation results. Then measure genuine quality-accepted end-to-end tokens/sec including I/O, eval, saves, recovery and failures. Run frozen NIS2 and held-out generation tests before any promotion. This draft does not claim any real full-model A/B execution, 44k/50k training speed, or quality gains.
