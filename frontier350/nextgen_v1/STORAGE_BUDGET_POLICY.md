# Neral checkpoint storage policy — 2026-09-17

## Verified facts and separate lineages

- Historical nextgen training documented checkpoints near 1.997 billion and 2.021 billion processed tokens (see issue #114). This is not evidence that each historical file remains present and SHA-verifiable **today**. Never substitute the separate 1.010-billion-token SAFE checkpoint as if it were the newest global training progress.
- The documented NIS2 exact free-answer result was 0/24 at the 2B diagnostic population. This does not by itself determine knowledge or imply that more pretraining improves free generation. Verify both checkpoints' actual bytes, identities and evaluation protocol before comparisons.
- Local disk scan before archive relocation: C: ~53.5 GiB free, D: ~674.8 GiB free. After safely relocating historical V2 and Throughput60kV3 with verified file counts/bytes and junctions: C: ~46.0 GiB free, D: ~682.1 GiB free. These are one-time measurements, not permanent guarantees. Never place active training on C: merely to free D:.

## Enforced now, but only for the explicit whole-model wrapper

`train_full_with_journal.py` imports `storage_budget.py` and refuses to begin unless its `--out` folder contains its own, directly named `resume.pt`, with no symlink/junction. Before Torch/model load and immediately before **every journaled checkpoint save**, it computes run directory size (counting hardlinked files once), requires the estimated extra write to fit under a fixed **40 GiB per-run limit**, and requires at least **200 GiB disk free after reserving the estimated extra write**. The estimated write reserve is twice current `resume.pt` bytes plus 1 GiB overhead. Errors stop the wrapper and do not delete previous checkpoints. CI runs synthetic storage and journal tests on hosted CPU only.

The 40 GiB ceiling and 200 GiB reserve cannot be relaxed via command-line flags. The script does not prune/rewrite/overwrite protected checkpoints, delete other runs or silently move folders to C:. Filesystem usage may change outside this process and the estimate is conservative, not a hard operating-system quota. Additional files written directly by unguarded scripts remain outside this policy.

**Important scope:** This is *not* a whole-project storage quota. Legacy V6/V8/V8.1 workflows, unrelated data caches, old generation archives, one-shot job checkouts, and experimental SFT output code are not retroactively protected by this wrapper. Do not run a legacy or auto-restarting training workflow until its own checkpoint and output paths have been audited and connected to an equivalent capacity guard.

## Next steps / authorization gates

1. Read-only inventory of actual surviving 1.997B/2.021B checkpoint files, SHA-256, training metadata, total project size per volume, remaining temporary checkouts and caches. No deletions or blind migration.
2. Independently freeze holdout v2 and complete human dataset/privacy/license review before any SFT training. Measure current checkpoint free generation against the same validated baseline and score with unchanged prompts.
3. If further training is justified, run only an explicit opt-in guarded short pilot. Monitor D: usage; stop when the per-run or disk reserve would be exceeded. Do not solve overflow by disabling checks or automatically deleting older checkpoints.
4. Any cleanup must be a separate reviewed operation with verified source/destination/checkpoint identity and a restorable copy. Keep the latest SHA-verified resumable checkpoint, the separately pinned SAFE checkpoint, critical optimizer/RNG/provenance metadata and independent evaluation artifacts until release review.

There was no GPU load, checkpoint deletion, training restart or model-weight change as part of implementing this storage policy.
