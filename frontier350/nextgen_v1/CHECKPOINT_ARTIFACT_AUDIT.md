# NextGen: checkpoint-byte audit (research-only)

The original 44k JSON contract could accept fabricated stage boundaries. `stage_ledger.py` now cross-checks a **separately pinned** three-stage JSONL ledger for continuous token counters, monotonic wall time and input/checkpoint identities. `checkpoint_artifacts.py` additionally streams **actual on-disk checkpoint bytes** to verify the four boundary SHA-256 values against the ledger. Neither test proves that CUDA ran, that the files contain valid Neral weights, that the sampler was correct, or that the checkpoints can resume exactly.

## Safe local invocation — NO GPU, no torch.load and no changes to checkpoint bytes

Only after a real isolated trainer has produced authentic immutable logs and four unique boundary files, retain the ledger SHA-256 outside the same untrusted run (for example in a separately signed or independently preserved artifact), and prepare a manifest such as:

```json
{"schema": 1, "checkpoints": [
  {"sha256": "<actual lowercase SHA-256 of baseline>", "filename": "baseline.pt"},
  {"sha256": "<actual lowercase SHA-256 of stage 1>", "filename": "stage1.pt"},
  {"sha256": "<actual lowercase SHA-256 of stage 2>", "filename": "stage2.pt"},
  {"sha256": "<actual lowercase SHA-256 of stage 3>", "filename": "stage3.pt"}
]}
```

Substitute actual 64-character lowercase hexadecimal hashes and use four distinct, ordinary filenames in a dedicated local directory. Run from the repository's `frontier350/nextgen_v1` directory:

```bash
python checkpoint_artifacts.py \
  --evidence /path/to/measured_evidence.json \
  --ledger /path/to/stages.jsonl \
  --expected-ledger-sha256 <SHA_FROM_INDEPENDENT_SOURCE> \
  --checkpoint-dir /path/to/isolated-checkpoints \
  --manifest /path/to/checkpoint_manifest.json \
  --out /path/to/new-checkpoint-audit.json
```

The audit refuses overwritten output files, missing/empty checkpoint files, symlink inputs, filename traversal, unexpected or duplicated manifest entries, mutated files during hashing and any mismatch with real bytes. It does not deserialize pickle files. The resulting `checkpoint_bytes_verified=true` means only that the currently readable files match the separately pinned ledger. `eligible_for_production` is **always false**.

## Blocking work after this audit

1. Determine the latest independently verified SAFE Neral-own checkpoint versus the V8.1 continuity-only candidate; do not silently use its 2.073B tokens as SAFE. Confirm trusted tokenizer and optimizer/RNG/sampler state. This audit cannot inspect checkpoint contents.
2. Implement a bounded A/B restart test in a separate local output directory using one identical data sequence; compare full weights, optimizer tensors, RNG and accepted-token history after uninterrupted versus interrupted training. A byte digest of two PyTorch archives is insufficient to establish numeric equivalence.
3. Instrument a real-data trainer to emit stage timing **from before input loading through validation, checkpoint hashing and recovery**. Do not derive the external SHA from the same untrusted ledger for a purported independent check.
4. Run complete frozen NIS2 and sealed holdout before calling any checkpoint better or promotable. Report genuine sustainable throughput, even if far under 44k. Do not run unattended multi-hour training or merge this draft PR.

The CPU CI uses tiny toy files and synthetic reports only, never user's local files or training. Do not commit real model checkpoints or datasets to GitHub by default.
