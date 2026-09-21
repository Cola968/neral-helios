# Phoenix training progress: actual local evidence only

`phoenix_training_progress.py` is a **read-only** monitor for the real bounded 350M weight-only pilot. It does not train, load model weights, copy or hash checkpoints, evaluate model answers, use the GPU, or upload anything. It reads only `metrics.jsonl` and `summary.json` from an already existing local pilot directory.

Run in a second PowerShell terminal from the checked-out experimental branch, with the *actual* pilot output directory in `$out`:

```powershell
python -B frontier350/nextgen_v1/phoenix_training_progress.py --run-dir "$out" --follow
```

Use the same command without `--follow` for a single JSON snapshot. `Ctrl+C` stops watching. See `PHOENIX_FIRST_TRAINING_RUN.md` for the explicit native preflight and **separately authorized** training command. This monitor never starts the trainer.

Metrics: new-only token count, approved pilot target, step count, most recently logged train/validation losses, elapsed invocation time and *observed wall-clock* tokens/s computed from token increment divided by elapsed invocation time. This includes checkpoint and evaluation overhead; it does **not** certify kernel training throughput. Training loss is not answer-quality evidence. The monitor prints no private manifest, prompt, file path or lineage.

The monitor rejects missing/contradictory JSONL receipts, unexpected step jumps, non-finite losses, changed approved targets or inconsistent completion counters. An incomplete final line during append is flagged and ignored. On an exact resume, `summary.json` from the previous invocation must not signal that the current invocation finished; a stale summary is ignored. A matching final summary reports `summary_reported_checkpoint_not_verified`: full checkpoint bytes, SHA, optimizer integrity, GPU telemetry and true model quality are **not** verified by this script. Only actual local training and independent evaluations can establish progress.

Verified in this development session: **13 synthetic isolated CPU unit tests passed** against small temporary files; no real 350M weights, owner filesystem or GPU was accessed. No new training tokens or quality gain were demonstrated. Repository-level hosted CI remains unverified at the time of this change; do not reproduce synthetic receipts as actual training logs.
