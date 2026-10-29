# Arctus350: SFT-supervision-efficiency audit (29. Oktober 2026)

**Goal:** Improve the *evidence used to decide the next training experiment*, not fabricate GPT-3/Giotto-level capabilities. The native Arctus350 architecture remains exactly **349,825,152 trainable parameters** with the locked original 16K BPE tokenizer. This audit performs no weight updates.

## Newly measured dimensions (on a supplied valid corpus and original tokenizer)

`frontier350/arctus350_supervision_audit.py` reuses the existing native `encode_messages` implementation, computes the exact causal `targets` (including `-100` masked user/context tokens) and counts:

- Eligible/ineligible examples by **train, val, test**, **language**, **domain** for the requested context width.
- **Assistant target tokens**, masked **context tokens**, proportion of tokens producing an assistant learning signal, and exact **terminal EOS** label count.
- Fails closed when any example has inconsistent assistant-target count, unmasked beginning-of-sequence context, malformed alignment, missing/multiple EOS, or padding tokens inside the loss targets.
- Reports aggregates **only**; does not print or persist private chat contents, IDs, token arrays, user messages or model weights.
- Counts overlong examples explicitly, so coverage is not silently inflated. It does **not** assume low supervision density is necessarily undesirable; the ratio is only a diagnostic that must be interpreted with quality metrics.

This module is kept distinct from:
- PR #180: held-out native paired-evidence promotion gates.
- PR #181: opt-in compact/bucketed training batch packing.
- PR #182: mathematical split contamination detection.

## CPU-only proof (synthetic)

The six TDD tests call the native SFT formatting routine with a **fake deterministic tokenizer solely for alignment checks**; do not interpret those token ratios as a measurement of the real Arctus corpus.

- RED: [GitHub Actions #40844852023](https://github.com/Cola968/neral-helios/actions/runs/40844852023) (missing audit implementation).
- GREEN: [GitHub Actions #40844937443](https://github.com/Cola968/neral-helios/actions/runs/40844937443) – 6/6 synthetic CPU tests passed. Tests cover masking, final EOS, multi-turn chat, language/domain/split reporting, ineligible lengths, invalid masks and determinism.

**Native training quality is NOT_MEASURED. No accuracy, NLL, EOS-generation rate or model improvement is claimed.**

## How to run a real data audit later (read-only; no weights needed)

With the *actual* clean NeralTalk split directory and the original Arctus 16K BPE tokenizer available locally:

```bash
python -m frontier350.arctus350_supervision_audit \
  --data /path/to/approved/neraltalk350 \
  --tokenizer /path/to/original/arctus16k-tokenizer.json \
  --seq 256 \
  --out /path/to/new/audit-report.json
```

The CLI checks the original tokenizer SHA-256 `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`, token mapping and original three-way split invariants. It **refuses overwriting existing reports**. Requires optional Python package `tokenizers`, but no PyTorch, GPU or model state.

Only use private data paths on the user's own machine; **do not upload private corpus, tokenizer or trained weights to GitHub or Hugging Face**. If the protected data/locked tokenizer are unavailable, report `NOT_MEASURED` rather than substitute a fake corpus.

## How this helps the next training experiment

After measuring real train/val examples, compare (a) assistant-to-context token ratio, (b) percent of usable records at context length 128/256/512, (c) EOS target coverage, and (d) per-language/domain distribution. Use those facts to select one controlled SFT ablation, with fixed dev+test holdouts and the exact same checkpoint. Do not infer model capability from synthetic padding reductions or supervised-token fraction.

**No paid HF Jobs, no PC/GPU run, no checkpoint promotion, no public release, no merge into main.** Financial ceiling remains 25 € direct project costs in total (exclusive of home electricity); no costs incurred by this change beyond the normal GitHub CI quota.
