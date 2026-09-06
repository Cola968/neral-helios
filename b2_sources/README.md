# Neral-50M B2 local source contract

`b2_sources/` is a **local data staging directory**. Raw JSONL source files and generated `.bin` files must not be committed. Only this contract and provenance metadata belong in Git.

## Goal

B2 is the first broader continued-pretraining stage after the Wikipedia-only B1 sweep. The initial target is **100,000,000 selected training byte/special tokens**. This target is a milestone, not a claim that 50M is fully pretrained.

A recommended first mix is:

| Domain | Planning share |
|---|---:|
| general clean language | 45% |
| educational explanations | 20% |
| science + mathematics | 10% |
| code / technical documentation | 8% |
| dialogue | 7% |
| astronomy / aviation / technical focus | 5% |
| structured high-quality text | 5% |

The manifest controls the actual source-level fractions. Shares must total exactly `1.0` and every source must have enough clean train-split tokens to fill its quota. The builder fails instead of silently distorting the mix.

## `manifest.json`

Example schema (replace all values with real, verified metadata):

```json
{
  "version": 1,
  "sources": [
    {
      "id": "example-general-en",
      "file": "example-general-en.jsonl",
      "sha256": "<64 lowercase hex characters>",
      "language": "en",
      "domain": "general",
      "target_fraction": 0.5,
      "license": "<exact license identifier>",
      "license_url": "<canonical license URL>",
      "provenance_url": "<dataset/source URL>"
    },
    {
      "id": "example-general-de",
      "file": "example-general-de.jsonl",
      "sha256": "<64 lowercase hex characters>",
      "language": "de",
      "domain": "general",
      "target_fraction": 0.5,
      "license": "<exact license identifier>",
      "license_url": "<canonical license URL>",
      "provenance_url": "<dataset/source URL>"
    }
  ]
}
```

Allowed languages: `de`, `en`.

Allowed domains: `general`, `education`, `science_math`, `code`, `dialogue`, `focus`, `structured`.

Do not invent a license. Do not stage a source until its redistribution/training terms and provenance are understood and recorded.

## JSONL source row

Each line is one complete document:

```json
{"id":"stable-document-id","text":"Complete source text...","title":"optional","url":"optional per-document attribution URL"}
```

The row `id` must be stable and unique within that source. Complete documents stay together across train/validation/test/holdout splitting.

## Build

```powershell
python prepare_pretrain50_b2.py --source-root b2_sources --out pretrain50_b2_data --target-train-tokens 100000000
```

The builder verifies source SHA-256 values before reading, applies deterministic filtering and deduplication, rejects exact frozen benchmark prompt contamination, creates deterministic 90/5/3/2 document-level splits, and records source/domain counts in `pretrain50_b2_data/manifest.json`.

Raw source documents and packed binaries remain private/local. The generated manifest can be copied into reports for audit without copying the corpus.
