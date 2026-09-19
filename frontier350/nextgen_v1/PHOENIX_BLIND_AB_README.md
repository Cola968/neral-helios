# Phoenix: private blind free-answer A/B (experimental)

`phoenix_blind_ab_audit.py` builds a **private, unscored**, per-question A/B review packet for the pinned unchanged 350M original checkpoint versus a genuinely distinct candidate. It does **not** load or train a model, generate answers, author an independent holdout, certify data rights, grade responses or demonstrate any improvement. Never publish the holdout, generated texts, answer key or reviews. There is no approved candidate training run at present.

## Inputs (all local JSON, UTF-8)

The separately and independently authored, manually checked, frozen holdout is an object with exactly `schema: 1` and `cases`, a list of 8–500 records, each with exactly `id`, `language` (`de` or `en`), `domain`, `prompt` and `rubric` (human grading criteria). Every ID and normalized prompt must be unique; both languages must occur. Eight publicly known development cases, including the existing Neral eight, cannot serve as an independent holdout. The script tests literal duplicates **inside** this file only; it cannot test semantic overlap or certify independence. Before using this tool, independently check the complete training/validation sources, original benchmark/development prompts, licensing, privacy, exact overlap and semantic contamination. Freeze and independently record the SHA-256 of the actual file bytes before model evaluation.

Supply separate baseline and candidate JSON objects with **exactly**: `schema: 1`, `checkpoint_sha256`, `tokenizer_sha256`, `holdout_sha256`, `max_new_tokens`, `decode` and `cases`. SHA strings are lower-case 64-character hex. Both `holdout_sha256` values must match the hash of the provided holdout bytes. Baseline checkpoint hash must be the immutable original `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`; candidate hash must be different. Record and verify actual checkpoint file hashes externally; this script cannot authenticate claims in JSON. `max_new_tokens` (1–512), complete `decode` parameter dict and tokenizer SHA must agree across runs. Compare using identical prompt formatting, tokenized input and generation cap; declare all decoding choices and verify they were actually used in both inference runs.

Each report's `cases` list must match the holdout ID **and order**, with exactly `id`, `prompt_sha256`, `input_ids_sha256`, `answer`, `generated_tokens`, `finish_reason`. `prompt_sha256` means SHA-256 of the exact UTF-8 prompt; `input_ids_sha256` means SHA-256 of the exact, identically serialized token-ID input, including chat/BOS markers. Recommended canonical serialization: `json.dumps(token_ids, separators=(',', ':'), ensure_ascii=False).encode('utf-8')` before SHA-256. Both runs must have identical input-ID digests for every case. `generated_tokens` is an integer from zero through the cap. `finish_reason` is `eos`, `max_tokens`, `role_boundary` or `repetition_guard`. These metadata fields must be captured from real inference, not filled with dummy values. The script cannot prove an inference process truly honored self-declared hashes or decoding settings.

## Operation (after real independent review and two actual inference reports)

Run on the user's local machine, never in public CI or a public runner. Use an existing private directory with path ending in `NeralPrivateAssets/phoenix`. Example (replace placeholders with actual local paths):

```bash
python frontier350/nextgen_v1/phoenix_blind_ab_audit.py --execute --holdout PATH_TO_FROZEN_HOLDOUT.json --baseline PATH_TO_REAL_ORIGINAL_OUTPUT.json --candidate PATH_TO_REAL_CANDIDATE_OUTPUT.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01
```

Creates two new, non-overwriting files: `phoenix_blind_pilot01.json` (only this packet goes to the independent human grader) and `phoenix_blind_pilot01.key.json` (secret per-case A/B mapping and actual checkpoint hashes; do not show the grader before scoring is frozen). Labels A/B are randomized separately per question. The packet includes prompts, rubrics and actual free answers; it is **private data**. The script logs only case count/hash and explicit `quality_improvement_proven: false`. It never auto-scores. A grader should assess each free answer against the rubric, freeze per-case decisions, and only then unblind the key. Assess correctness, nonanswers, hallucinations and unsafe outputs with a predetermined rubric. A small pilot cannot establish broad capability.

## Verify contracts without model/GPU

From `frontier350/nextgen_v1` run:

```bash
python -m unittest -v test_phoenix_blind_ab_audit.py
```

Seven synthetic local tests exercise masking/key correspondence and rejection of changed inputs, altered holdout, identical checkpoints, changed decoding, repeated prompts and boolean token counts. Passing them **does not** establish independent data, model quality, true 350M GPU feasibility, throughput, or checkpoint recovery. Keep the training approval gate closed, PR #118 draft and issue #119 open until all real evidence is supplied.
