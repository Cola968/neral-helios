# Neral Hugging Face data policy

This document records the Hugging Face datasets inspected for Neral and the corresponding project decision. The executable source of truth is `hf_source_policy50.py`.

## Current I1 sources

### OpenAssistant/oasst2

- purpose: human instruction-tuning data
- license: Apache-2.0
- pinned revision: `179dd21fc55192153d94adb0e0ce8f69e222bf75`
- pinned ready-message archive SHA-256: `a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751`
- upstream ready export: 135,174 messages; the dataset includes German and English
- Neral filter: `review_result == true`, `deleted == false`, `synthetic == false`, no `model_name`, matching-language human prompter -> assistant pairs, deterministic best-ranked assistant per prompt
- user IDs and other account metadata are not copied into the Neral SFT JSONL

OASST2 replaces OASST1 in the I1 D3 mixture rather than being stacked on top of it. This avoids deliberately duplicating the older OpenAssistant release. OASST1 remains registered for reproduction of historical runs.

The 20k I1 SFT target keeps a 30% human share: 24% English OASST2 and 6% German OASST2. The remaining 70% is Neral's deterministic project-generated German/math/code/reasoning curriculum. This doubles the human German quota relative to the previous I1 plan without increasing the amount of external data.

## Public benchmark holdouts

The following datasets are intentionally **not training data**:

- `openai/gsm8k` — MIT, crowdsourced grade-school math; reserved for a future external math evaluation.
- `google-research-datasets/mbpp` — CC-BY-4.0, crowd/expert Python problems; reserved for a future external code evaluation.
- `deepmind/code_contests` — CC-BY-4.0 competitive-programming problems; reserved for future code evaluation.

`hf_source_policy50.py` fails closed if any of these IDs are passed to a training-source policy check. This makes future public benchmark claims more defensible.

## Future candidates, not I1 inputs

### HuggingFaceTB/finemath

FineMath is an ODC-By mathematical educational web corpus. It is a strong candidate for a later I2 pretraining experiment, but it is deliberately blocked for the current campaign because adding it after B3 began would invalidate the frozen I1 curriculum. Before I2 it should receive an independent contamination audit against every Neral and public math benchmark.

### HuggingFaceTB/stack-edu

Stack-Edu is potentially valuable high-quality educational code. It is not approved yet because Neral needs a file-level/source-license policy for code corpora derived from large collections of repositories. Do not treat a high-level dataset description as sufficient license clearance.

### HuggingFaceTB/smollm-corpus

The corpus includes large model-generated components. It is excluded under the current Neral policy that avoids using external-model-generated training content for the core model.

### allenai/tulu-3-sft-mixture

The mixture combines human and machine-generated data and contains subsets with different, including non-commercial, licenses. It is blocked for Neral's current training pipeline.

### codeparrot/codeparrot-clean

The dataset is useful Python code, but the project does not currently have sufficiently explicit source-license handling for it. It remains blocked pending a dedicated licensing and provenance review.

## Existing pretraining sources

Neral's existing approved web sources remain `HuggingFaceFW/fineweb`, `HuggingFaceFW/fineweb-2` and `HuggingFaceFW/fineweb-edu`. Their use is already captured by the B2/B3 provenance manifests. This policy does not change an in-progress B3 checkpoint or corpus.

## Rule for future campaigns

A dataset is not admitted because it is popular or improves another model. It must have:

1. an explicit role (pretrain, human SFT, or evaluation holdout),
2. recorded license/provenance,
3. a pinned revision and file hashes before weight-changing training,
4. deterministic cleaning and split logic,
5. contamination screening against all frozen Neral and public benchmark suites,
6. a fail-closed policy if any of those requirements cannot be demonstrated.
