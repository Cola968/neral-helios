# Arctus350: offline near-duplicate and split-leakage audit

**Draft PR #208, stacked on the billion-scale data staging Draft #205.**
This is an inspectable offline quality-control tool, **not** a trained
checkpoint, a license grant, or a release certification.

## Why it matters

The existing `arctus_billion_data.py` deliberately deduplicates exact
normalized-text SHA-256 values with a disk-backed database, but marks
`data_is_training_ready: false`. Two almost identical articles can have
different SHA-256 hashes. Documents copied across the train and validation
splits can artificially improve validation loss.

`arctus_billion_audit.py` scans a **previously staged** source directory
from PR #205. Nothing is downloaded. It does not contact Hugging Face or
train Arctus350. It does not modify any staged document or manifest.

## Usage

Default invocation is a no-cost, read-only plan:

```bash
python arctus_billion_audit.py
```

After an operator explicitly stages data using the existing
`arctus_billion_data.py --execute --allow-download ...` workflow, a
local audit can examine those JSONL shards (this never allocates a GPU):

```bash
python arctus_billion_audit.py --corpus /trusted/NERAL/approved-staged-data
```

Optionally write a **new** audit report, with a path outside the
training corpus; an existing report is never silently overwritten:

```bash
python arctus_billion_audit.py \
  --corpus /trusted/NERAL/approved-staged-data \
  --report /trusted/NERAL/reviews/arctus-audit-01.json
```

The report includes manifest SHA-256, total examined documents,
duplicate statistics (exact, within split, across train/val/test/holdout),
and an explicit `training_ready: false` result even if no duplicates
are found. It now also includes **up to 20 actionable `example_pairs`**
with the original and duplicate shard names (`train-*.jsonl`, `val-*.jsonl`,
`test-*.jsonl` or `holdout-*.jsonl`) and 1-based row numbers. No raw document text, URLs or
source row IDs are copied into the report. Set `--sample-limit 100` to
inspect more cases (hard maximum 1,000, to bound output).

Each example has `kind` (`exact`, `within_split_near` or
`cross_split_near`), `original:{shard,line}` and
`duplicate:{shard,line}`. `example_pairs_are_exhaustive=false`
means the bounded examples are *not* a complete quarantine list:
the aggregate counts still cover all checked pairs. The sample
locations enable a manual review/exclusion procedure before a new
manifest is prepared, **without modifying the staged source data**. A pass means **only** that this particular audit found no
violations. The report is derived entirely from actual inspected data,
not an estimate of future corpus size.

### Invariants

- The existing staged `manifest.json` must be schema version 1,
  `NERAL-16K-BPE`, `trained_tokens=0`, `training_started=false`,
  `data_is_training_ready=false`.
- Every `train-*.jsonl`, `val-*.jsonl`, `test-*.jsonl` and
  `holdout-*.jsonl` shard produced by the existing builder must be listed
  exactly once with a valid path, SHA-256 and byte/token/document counts.
  Legacy `valid-*.jsonl` shards remain supported for compatibility, but
  `val` and `valid` may not coexist in one corpus. Every partition's token
  totals must reconcile with the original manifest.
- Source license, revision, language and domain must match the pinned
  manifest entries; normalized text SHA-256 must match the actual row.
- Disk-backed SQLite stores the digest + 64-bit word-trigram SimHash;
  five LSH bands (13/13/13/13/12 bits) make Hamming distance <=4
  candidates visible. Queries are capped; saturated buckets raise
  **inconclusive**, rather than returning a falsely clean report.
- The temporary index is removed after the audit. The original corpus
  is not modified. No checkpoint or private model weights are loaded.

### What it cannot prove

- SimHash is a conservative syntactic near-duplicate heuristic, **not**
  complete semantic similarity detection. A good paraphrase may
  be missed. An overly templated document may trigger a false positive.
- This is not a replacement for frozen benchmark contamination scans,
  long-form/semantic leakage checks, licenses/usage-rights review,
  sampling inspection, or pretraining-vs-SFT separation. It also does
  not guarantee cross-corpus overlap is eliminated where earlier
  historical indexes lack near-duplicate signatures.
- The library README/license needs to be read and verified for each
  exact pinned revision, not merely inferred from an ODC-By tag.
  Original project source policy continues to apply.
- A good data-quality report **cannot** establish that Arctus350 is
  smarter. Native checkpoint quality requires matched held-out
  evaluations and actual GPU training.

## Reproduction

```bash
python -m pip install pytest
python -m pytest -q tests/test_arctus_billion_audit.py
python arctus_billion_audit.py
```

All tests use **synthetic text only**. No GPU, Hugging Face job, paid
service or user checkpoint access is required.
