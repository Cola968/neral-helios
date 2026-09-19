# Phoenix private A/B — blind free answers and human scoring (experimental)

**No real model training, new 350M weights or quality gain resulted from this work.** The tools build a private unscored A/B comparison from externally generated answers, then aggregate separately frozen human correctness grades. They do not author an independent holdout, audit training data rights/contamination, generate model answers, inspect actual weight files, grade answers, establish human grader independence, or prove improvement. Never publish holdout text, packet, answers, grading notes or secret key.

## 1. Obtain independent evidence before running

Independently author a frozen, manually checked bilingual UTF-8 JSON holdout with shape `{"schema":1,"cases":[...]}` and 8–500 cases, each containing *exactly* `id`, `language` (`de` or `en`), `domain`, `prompt`, `rubric`. Both languages must occur and IDs and normalized prompts must be unique. Earlier public Neral development questions cannot count as an independent holdout. Before use, human-review training/validation source licensing, privacy and content and check exact **and semantic** contamination across all training, validation and development materials. Independently preserve the holdout's actual SHA-256. The code does not perform or certify these external reviews.

Produce two **actual inference** JSON reports, original baseline and distinct candidate, with exact fields `schema` (integer 1), `checkpoint_sha256`, `tokenizer_sha256`, `holdout_sha256`, `max_new_tokens`, `decode`, `cases`. Each digest must be a 64-character lowercase SHA-256. Baseline actual checkpoint hash must equal `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`; candidate SHA must differ; both must use and independently verify the same original Neral tokenizer, frozen holdout bytes, full decoding config and generation limit. Metadata supplied in JSON **is an unverified declaration**; hash the actual local model/tokenizer assets independently and verify actual decoding behavior before claiming an authentic evaluation.

Each report must list cases in exact holdout order with exactly `id`, `prompt_sha256`, `input_ids_sha256`, `answer`, `generated_tokens`, `finish_reason`. The prompt hash is over exact UTF-8 prompt bytes. Hash identical canonical serialized input token IDs, including chat/BOS markers, e.g. `json.dumps(token_ids, separators=(',', ':'), ensure_ascii=False).encode('utf-8')`. Report the real free answer, 0..cap generated tokens and finish reason `eos`, `max_tokens`, `role_boundary` or `repetition_guard`. Case prompt, input IDs, tokenization and decode must match both models; scripts check declared consistency but cannot prove inference occurred.

## 2. Create private blind packet

Execute only on the owner's machine, after independent approval and with an existing directory ending in `NeralPrivateAssets/phoenix`:

```bash
python frontier350/nextgen_v1/phoenix_blind_ab_audit.py --execute --holdout PATH_TO_FROZEN_HOLDOUT.json --baseline PATH_TO_REAL_ORIGINAL_OUTPUT.json --candidate PATH_TO_REAL_CANDIDATE_OUTPUT.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01
```

This creates two distinct non-overwriting files: `phoenix_blind_pilot01.json` for the human grader and `phoenix_blind_pilot01.key.json` as a **secret** per-case A/B assignment. The key contains a SHA-256 commitment to the exact packet bytes and hashes of each A/B answer. A key from another randomization of the same holdout or changed packet answers is rejected during unblinding. SHA is not an identity signature; the key cannot prove an evaluator remained blind. Never show the key to the grader until grades are frozen.

## 3. Freeze human grades, then unblind

The independent human grader receives only the packet and preregistered rubric, and creates `grades.json` with exactly `schema: 1`, `packet_sha256` (hash of the exact created packet bytes) and `items`. Each item must appear **exactly once in packet order** with `id`, Boolean `A_correct`, Boolean `B_correct`, and nonempty `notes` (1–2000 characters of actual rubric-grounded reasoning). Freeze grades before revealing the mapping; preserve the grade file SHA separately. These records are human-provided, not software-generated truth.

Run locally after grades are saved:

```bash
python frontier350/nextgen_v1/phoenix_blind_ab_unblind.py --execute --packet D:/NeralPrivateAssets/phoenix/phoenix_blind_pilot01.json --grades PATH_TO_FROZEN_GRADES.json --secret-key D:/NeralPrivateAssets/phoenix/phoenix_blind_pilot01.key.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01
```

The create-only private `phoenix_scored_pilot01.json` includes original and candidate counts, both-correct, both-wrong, original-only, candidate-only, paired net correct count and language/domain subtotals. Logs contain aggregates only. `quality_improvement_proven` deliberately remains false: a positive count difference or small synthetic pilot is not proof of general ability. Do not choose new checkpoints by repeatedly inspecting a frozen independent holdout.

## 4. Verification and current limitations

From `frontier350/nextgen_v1`:

```bash
python -m unittest -v test_phoenix_blind_ab_audit.py test_phoenix_blind_ab_unblind.py
python -m py_compile phoenix_blind_ab_audit.py phoenix_blind_ab_unblind.py test_phoenix_blind_ab_audit.py test_phoenix_blind_ab_unblind.py
```

**19 isolated synthetic local tests passed**, including real packet-writer CLI serialization/hash integrity, private file workflow for human score aggregation, overwrite refusal, altered answers, unrelated key reused on the same holdout and mismatch rejection. GitHub-hosted Python syntax workflow has reported failure with unavailable job steps/logs; hosted CI is **not green**. Neither model training nor an actual owner-PC 350M inference test was run here. Required training reviews, approved pilot, real CUDA memory/temperature/disk/throughput, full-size recovery and a truly independent blinded free-answer evaluation all remain outstanding. Keep PR #118 and #120 draft, and issue #119 open.
