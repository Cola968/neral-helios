# Neral Data Scale V2 — 2026-09-08

## Objective

Increase useful training-data throughput by orders of magnitude without turning raw token count
into the optimization target. Neral keeps its own weights and byte vocabulary. External public
corpora are data sources only; no foreign pretrained weights are loaded.

The program is deliberately staged because Neral byte-tokens are not directly comparable to
BPE/SentencePiece token counts used by most public LLM reports.

## Scale ladder

| Generation | Parameters | Data target | Decision rule |
|---|---:|---:|---|
| Neral 50M I1 | 50,012,976 | 500M cumulative byte tokens + guarded math/SFT | finish current frozen campaign first |
| Neral 50M H1 | 50,012,976 | 1B cumulative byte tokens | only if I1 remains compute/data limited and frozen quality improves |
| Neral 50M H2 research | 50,012,976 | 2B cumulative byte tokens | continue only while held-out CE/BPB and NIS2 slope remain positive |
| Neral Orion 100M | ~100M | 2B initial, expandable to 5B byte tokens | staged from-scratch pretraining; benchmark every major tranche |
| Neral ~250M | ~250M | >=5B byte tokens | requires Orion scaling evidence |
| Neral ~500M | ~500M | >=10B byte tokens | long-term primary data-scale milestone |
| Neral ~1B | ~1B | >=30B byte tokens research target | only after compute/tokenizer architecture is re-evaluated |

No stage is promoted merely for consuming its token budget.

## Candidate source pool

### Tier A — preferred general/educational pretraining

1. **HuggingFaceFW/fineweb-2**
   - multilingual high-quality web corpus, including German;
   - dataset card reports ODC-By 1.0;
   - use German and English only for the current Neral curriculum;
   - immutable Hub revision must be recorded before admission.

2. **HuggingFaceFW/fineweb-edu**
   - English educational web corpus;
   - dataset card reports ODC-By;
   - useful for explanation/science/general knowledge quality;
   - immutable Hub revision must be recorded before admission.

3. **Wikimedia/Wikipedia DE+EN**
   - keep as a smaller encyclopedic anchor, not the dominant future corpus;
   - existing project provenance rules continue to apply.

### Tier B — targeted capability data

4. **HuggingFaceTB/finemath**
   - mathematical educational content;
   - dataset card reports ODC-By;
   - candidate only after the repository's existing contamination/licensing audit passes;
   - never mix benchmark holdouts into training.

5. **OpenAssistant/oasst2**
   - human-feedback conversation trees; Apache-2.0 per dataset card;
   - instruction tuning only, not bulk language pretraining;
   - German/English human-only paths preferred.

6. **Code corpus**
   - The Stack v2 is useful but gated and license-sensitive;
   - it remains fail-closed until source-level license/access rules are implemented;
   - before that, use explicitly admissible project/code/documentation sources only.

### Excluded by default

- benchmark/evaluation holdouts;
- datasets with unresolved license/provenance;
- duplicate or near-duplicate documents already present in train/val/test/holdout;
- private/PII-bearing material;
- low-quality SEO/spam/template pages;
- model-generated synthetic corpora unless separately approved and measured;
- any foreign model weights, LoRAs, learned embeddings or distillation logits.

## Proposed large-scale mixture

This is a *starting allocation*, not a frozen recipe:

- 30% German FineWeb2 high-quality documents
- 25% English FineWeb-Edu
- 20% English FineWeb2 high-quality documents
- 10% FineMath after admission
- 5% encyclopedic/science anchor (DE/EN Wikipedia and admissible scientific text)
- 10% code/technical text once the code source passes license/provenance gates

For a stage where FineMath or code is not admitted, redistribute those shares to Tier-A sources;
do not silently substitute unreviewed data.

## Data-pipeline requirements

For multi-billion byte-token campaigns the existing small-corpus workflow is insufficient. V2
must use:

1. **streaming/sharded ingestion** rather than loading the entire corpus in RAM;
2. **document-level filtering before tokenization**;
3. **exact hash dedup + scalable near-dedup** on normalized documents;
4. **immutable source revision + shard/checksum manifests**;
5. **train/validation/test/holdout split before binary packing**;
6. **packed uint16 byte-token shards** with document boundary metadata;
7. **large sequential reads / memory mapping** during training;
8. **CPU preparation decoupled from the GPU training loop**;
9. **prefetch/pinned-memory staging** so the RTX 4070 does not wait on Python/PCIe;
10. **fail-closed resume signatures** covering corpus, ordering and critical hyperparameters.

## Quality filters

Every source should expose counts for accepted/rejected documents and rejection reasons. Minimum
filters: encoding/control characters, boilerplate/template repetition, contact/secret patterns,
very low alphabetic/content ratio, pathological repetition, exact duplicate, near duplicate,
benchmark contamination, document length and malformed markup.

Long documents should be packed without allowing training attention to cross source-document
boundaries unless an explicitly tested packing format introduces separators and masks.

## Token accounting

Always report both:

- Neral byte/special tokens actually processed;
- source bytes/documents and language/domain mixture.

Never compare Neral's byte-token budget one-to-one with public subword-token training budgets.

## Promotion gates per tranche

At 0.5B / 1B / 2B / 5B / 10B-class milestones as applicable:

- held-out DE/EN CE, perplexity and BPB;
- NIS2 model-only score and components;
- RawMath ID/transfer/OOD;
- code and reasoning suites;
- context/needle tests;
- EOS/repetition/UTF-8 generation health;
- forgetting against frozen previous-corpus validation;
- tokens/sec, GPU utilization proxy, peak VRAM, wall-clock tokens/hour.

Training continues only if the capability/validation trend justifies the next tranche.

## Hugging Face status checked 2026-09-08

The connected Hugging Face account is authenticated but is not a Pro account. Hugging Face Jobs
therefore is not treated as available paid training compute in this plan. Hugging Face is used
for dataset/research discovery and immutable source auditing; the primary training path remains
the user's self-hosted RTX runner unless compute availability changes.
