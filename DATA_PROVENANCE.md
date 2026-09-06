# Training data provenance

## Supplied Neral handoff

The original 20M checkpoint and corpus were supplied by the project owner. Historical provenance before that handoff is not independently known. The duplicated original corpus is retained locally for auditing. It is not uploaded as raw data.

## OpenAssistant/oasst1

- Source: https://huggingface.co/datasets/OpenAssistant/oasst1
- Revision: `fdf72ae0827c1cda404aff25b6603abec9e3399b`
- License: Apache-2.0; original license is included under `third_party/oasst1/`.
- Processing: approved, source-labelled non-synthetic root prompt/rank-0 response pairs, German/English, quality and content filters, complete answers up to 1024 byte tokens, normalized deduplication and tree-level splits.
- Attribution: OpenAssistant contributors. Authorship and quality are source metadata, not independent guarantees.
- Used in historical 20M/40M dialogue pilots, not the 50M language-pretraining pilot.

## Wikimedia/Wikipedia

- Source: https://huggingface.co/datasets/wikimedia/wikipedia
- Snapshot configurations: `20231101.de`, `20231101.en`.
- Dataset-card repository revision: `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
- License as stated by the dataset card: CC-BY-SA-3.0 and GFDL. The language pilot uses the CC-BY-SA-3.0 attribution path: https://creativecommons.org/licenses/by-sa/3.0/legalcode
- Contributors are credited through each article's source URL in the local document manifest. Source metadata and row-response SHA-256 hashes are retained. The full legal-code text download was unavailable from this environment; its exact license version and canonical link are recorded instead.
- Sample: 5,000 article records downloaded; 4,120 retained after deterministic filtering. 880 rejected (878 insufficient clean prose, 2 contact/secret patterns).
- Filters: encoding/control characters, contact/secret patterns, paragraph completeness and length, markup, alphabetic ratio, repeated characters and repeated paragraphs. No claim of exhaustive PII detection or manual verification of every statement.
- Deduplication: exact normalized hashes plus approximate bottom-64 word-five-shingle comparison. No duplicate was found in this bounded sample; the heuristic is not exhaustive semantic duplicate detection.
- Full articles grouped into train/validation/test/holdout before binary packing. Cross-language translations are not independently clustered.
- Training corpus: 3,736 articles and 21,915,506 byte/special tokens. German and English are sampled at 55/45 batch probability; actual processed counts are reported per run.
- No dedicated code corpus in this initial language pilot. It is encyclopedia text, not a broad finished conversational corpus.

Raw source documents, private project data and training binaries are excluded from the Git repository. Preparation scripts, source metadata, license notices and filtered counts are included. No external model weights, LoRAs, learned embeddings or hidden model-inference calls are used.
