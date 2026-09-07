# Neral-50M B2 local source contract

`b2_sources/` is a **local data staging directory**. Raw JSONL source files and generated `.bin` files must not be committed. Only this contract and provenance metadata belong in Git.

## Goal

B2 is the first broader continued-pretraining stage after the Wikipedia-only B1 sweep. The initial target is **100,000,000 selected training byte/special tokens**. This target is a milestone, not a claim that 50M is fully pretrained.

A recommended long-run mix is:

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

## Automatic first bootstrap

For the first reproducible B2 100M milestone, `bootstrap_b2_sources.py` stages a deliberately simpler broad-data mix automatically:

- 35% English FineWeb general text (`v1.4.0`),
- 40% German FineWeb2 `deu_Latn` (`v2.1.1`),
- 20% English FineWeb-Edu (`v1.4.0`),
- 5% positively reviewed, non-synthetic **English** OpenAssistant dialogue from the pinned OASST1 ready export.

German conversational behavior is supplied later by Stage-D DE/EN instruction tuning; the B2 dialogue source is English-only so its manifest language metadata remains exact.

The automatic bootstrap intentionally leaves license-heterogeneous public code datasets out. A dedicated code stage can later add code with per-source license/provenance controls instead of weakening B2 provenance.

Install the optional streaming dependency and stage the source files:

```powershell
python -m pip install -r requirements-data.txt
python bootstrap_b2_sources.py --out b2_sources
python prepare_pretrain50_b2.py --source-root b2_sources --out pretrain50_b2_data --target-train-tokens 100000000
```

The script pins dataset revisions, streams only a bounded amount of data, hashes every locally written source file, and writes the exact manifest consumed by the normal B2 builder. FineWeb, FineWeb2 and FineWeb-Edu are declared as ODC-By-1.0 data; their dataset cards also state that Common Crawl Terms of Use apply. OASST1 is recorded as Apache-2.0. Preserve attribution/provenance in any later model release documentation.

The downstream B2 builder still performs its own filtering, normalized exact deduplication, near-duplicate screening, document-level split and frozen-benchmark contamination screen. The bootstrap is **not** a replacement for those checks.

## `manifest.json`

Example schema (replace all values with real, verified metadata when adding a custom source):

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

The builder verifies source SHA-256 values before reading, applies deterministic filtering and deduplication, rejects exact frozen benchmark prompt/continuation contamination, creates deterministic 90/5/3/2 document-level splits, and records source/domain counts plus output hashes in `pretrain50_b2_data/manifest.json`.

Raw source documents and packed binaries remain private/local. The generated manifest can be copied into reports for audit without copying the corpus.
