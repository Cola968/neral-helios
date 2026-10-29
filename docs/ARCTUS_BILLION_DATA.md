# Arctus350: billion-token data pipeline (preparation only)

**State:** GitHub Draft; offline tests only. This is NOT a new model checkpoint,
training run, quality improvement measurement or approval to use the local GPU.

## Why more raw data alone is not enough

Issue #141 already documents a checked Arctus350 checkpoint reporting
`2,073,161,216` processed training tokens. More text does not imply better
answers: existing 1B/2B native checkpoints still lack a complete frozen
capability comparison. Issue #179 remains the quality/evaluation gate.

This pipeline builds a *reviewable, on-disk training-data candidate* using
existing B3-approved sources and the user's **own original 16K BPE tokenizer**.
It does not include or load Qwen weights, change Arctus weights, or train.

## Data source plan (for 1 billion unique selected train BPE tokens)

| Share | Dataset | Language | Domain |
| --- | --- | --- | --- |
| 40% (400M) | HuggingFaceFW/fineweb | English | broad web |
| 40% (400M) | HuggingFaceFW/fineweb-2 / deu_Latn | German | broad web |
| 20% (200M) | HuggingFaceFW/fineweb-edu | English | educational web |

This **reuses** `bootstrap_b3_sources.py`, `hf_source_policy50.py`,
and `prepare_pretrain50_b2.py` cleaning/contamination utilities; it
does not create an independent license registry. Only reviewed pretrain
datasets are admitted. Actual upstream tags are first resolved into
immutable 40-digit Git commit SHAs before streaming. Every output shard
has its own SHA-256, dataset revision, source ID, language, domain, license,
and split. FineWeb-family data have ODC-By terms and Common Crawl
upstream obligations: audit licenses and attribution before publication.

Large-scale near-duplicate detection and cross-campaign B2/B3 exclusion
**are not yet complete**. This is why the emitted manifest deliberately
contains `data_is_training_ready=false`. No training step consumes it
automatically. Exact normalized duplicates are rejected with SQLite
on disk; embedded frozen benchmark literals are rejected via the existing
best-effort literal-screening helper. It is NOT a general semantic-leakage
or complete PII detector.

## Default: safe offline plan

```powershell
python arctus_billion_data.py --target-train-tokens 1000000000
```

This prints quotas. It does NOT touch files or network. The target is a
**goal**, not an already-downloaded or trained amount.

## Explicit future local acquisition only (requires approval, disk, bandwidth)

```powershell
python -m pip install -r requirements-data.txt
python -m pip install tokenizers
python arctus_billion_data.py --execute --allow-download --target-train-tokens 1000000000 --tokenizer "C:\\path\\to\\your\\verified\\tokenizer.json" --out ".data\\arctus350-1b"
```

This can involve downloading multiple GB, substantial disk usage,
many CPU hours and Internet bandwidth, depending on corpus quality and
tokenizer speed. The command is **not run by CI or by this PR**.
It verifies the tokenizer against the project's locked digest, resolves
source tags to git SHA, streams with finite-size output shards, and records
the number of actually selected BPE training tokens. Incomplete
ingestion cannot publish `manifest.json`, and existing output directories
are never overwritten. If an incomplete directory remains, inspect it
before selecting a new destination; do not silently resume or treat its
unsealed shards as valid.

## Gates before weight updates

1. Verify per-shard SHA-256, immutable upstream revisions, ODC-By
   attribution and dataset card/underlying content usage rights.
2. Audit additional near-duplicates and overlap against historical
   B2/B3/Arctus checkpoint source indexes.
3. Freeze held-out evaluation cases and check data contamination.
4. Benchmark *existing* 1B, 2B and best SFT native checkpoints; determine
   whether more pretraining, targeted SFT or training recipe work is
   the higher-value next step.
5. After separate approval only: start a small GPU ablation with a fixed
   token budget, capped thermals, no checkpoint overwrite and
   honest comparisons. Do not infer a GPT-3-quality model from 1B data.

## Offline verification

```bash
python -m pytest -q tests/test_arctus_billion_data.py tests/test_hf_data_policy50.py tests/test_b2_data.py
```

References:
- Existing policy: `HF_DATA_POLICY.md`
- Existing native research: issue #179
- Verified native 2B checkpoint provenance: issue #141
- https://huggingface.co/datasets/HuggingFaceFW/fineweb
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
