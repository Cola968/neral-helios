# Arctus350: CPU-only sampler cost audit (29.10.2026)

Research issue: #179. Draft dependency: integrated research PR #186.
This is **not** a model evaluation, GPU benchmark, training result or
evidence of increased intelligence.

## Problem and method

The existing native SFT trainer offers the historical `bucketed` sampler and
an opt-in `compact` sampler (PR #181). Prior tests established less synthetic
padding for compact batching, but did not quantify the attention-shape proxy
under the **same complete set of examples**.

`arctus350_padding_cost_audit.py` imports the actual production
`epoch_order` and `compact_epoch_order` functions and, for each deterministic
seed, compares the same samples with exactly one visit per epoch:

- **Padded token count:** sum(batch_size × maximum sequence length per batch).
- **Attention proxy:** sum(batch_size × maximum sequence length² per batch).
- **Real token count:** sum(original lengths), identical in both strategies.

The quadratic proxy is an upper-bound-style shape indicator for a conventional
padded attention implementation. It is **not** a measured CUDA FLOP count,
wall-clock time, FlashAttention kernel cost, optimizer efficiency or VRAM peak.
Compact batching changes order and could affect generalization. No sampler
is automatically selected.

The audit intentionally requires at least two **full** batches and rejects
partial epoch tails, which could otherwise bias comparisons. Results include
median and min/max reduction over 1–16 deterministic seeds, plus an input
fingerprint. A negative reduction is reported honestly if compact loses.

## Usage (offline only)

Create a **local** UTF-8 JSON file containing only integer token lengths,
not source prompts, labels, IDs or checkpoint metadata:

```json
{"lengths":[32,64,96,128,48,72,120,144]}
```

Run from the repository root on an explicitly authorized offline machine:

```bash
python -m frontier350.arctus350_padding_cost_audit \
  --lengths-json /local/lengths-only.json \
  --batch-size 2 --bucket-size 8 --seeds 4 \
  --out /local/new-aggregate-report.json
```

The output file must not exist. The CLI accepts no raw text or private
metadata and emits **only aggregate costs**, a length-vector SHA-256 and
limitations. Do **not** upload real token-length vectors or raw datasets
to GitHub CI, Issues, external tools or cloud runners. CI uses synthetic
in-repository fixtures only. A fingerprint is not a privacy guarantee for
tiny or guessable length arrays; publish aggregate reports only after review.

## Scope and next decision

This is a free CPU-only **preflight**. To decide whether `compact` is genuinely
better for Arctus350, a later separately authorized local experiment must
compare (1) the actual SFT length distribution, (2) measured GPU time/VRAM,
and (3) unchanged frozen native validation cases. No PC/GPU run or model
weight change is authorized by this document. No TEST/NIS2 tuning.

## Paired-seed risk evidence (14:00 follow-up, Draft PR #200)

In addition to medians, the report now exports paired `compact_better`,
`equal`, and `compact_worse` seed counts separately for padded tokens and
the theoretical attention proxy, an attention-proxy percentage range, and
`compact_nonregressing_in_all_seeds`. These are **descriptive outcomes for
the chosen deterministic seeds only**, not statistical confidence intervals
or evidence of faster real training. An improvement in median alone does not
justify switching samplers; negative seed outcomes must remain visible.
The length-only input/privacy restrictions and no-training policy remain
unchanged.
