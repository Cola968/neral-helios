# Hugging Face discovery audit — 2026-09-08

This note records Hub metadata inspected during the Neral M2.1 campaign. It is research/provenance evidence only. No foreign model weights are imported and no new corpus is inserted into the already-running M2.1 curriculum.

## Live-inspected datasets

### `HuggingFaceFW/fineweb-edu`

- Hub license metadata: ODC-By.
- English educational web corpus; dataset card describes roughly 1.3T tokens.
- Educational filtering uses a classifier whose training labels were produced with an external instruction model. Neral therefore records the distinction between **web-found training text** and **model-assisted filtering metadata**; this is not treated as permission to introduce generated training rows.
- Status in Neral: already approved pretraining source represented in the B3 provenance chain.

### `HuggingFaceFW/fineweb-2`

- Hub license metadata: ODC-By.
- Multilingual web corpus spanning more than 1,000 languages, including German.
- Status in Neral: already approved; German data is represented in the existing B2/B3 provenance manifests.

### `OpenAssistant/oasst2`

- Hub license metadata: Apache-2.0.
- Human-feedback conversation-tree dataset with German and English content.
- Status in Neral: approved human SFT source. Neral keeps only reviewed, non-deleted, non-synthetic human prompter/assistant pairs and pins its upstream revision/archive hash separately.

### `HuggingFaceTB/finemath`

- Hub license metadata: ODC-By.
- Mathematical educational web corpus; dataset card exposes 34B/54B-token variants and describes model-assisted quality filtering.
- Status in Neral: **quarantined future candidate**. It is not M2.1 training data. Before any weight-changing experiment it must pass the fail-closed candidate admission audit, including immutable revision pinning, file hashes, deterministic cleaning provenance and zero detected overlap against all reserved benchmark screens.

## Architecture research only

`HuggingFaceTB/SmolLM2-135M` and `HuggingFaceTB/SmolLM2-135M-Instruct` were inspected only as public small-language-model references. Hub metadata reports about 134.5M parameters, Llama-family architecture and Apache-2.0 licensing. No checkpoint, tokenizer, embedding, LoRA or other learned parameter from these models is imported into Neral.

## Admission rule

`hf_candidate_admission50.py` is the additional fail-closed pre-admission layer for future candidates. Passing it does **not** change `hf_source_policy50.py` and does not authorize training. A future corpus must still receive an explicit policy change after manual review. Public evaluation holdouts (`openai/gsm8k`, `google-research-datasets/mbpp`, `deepmind/code_contests`) remain permanently excluded from training.
