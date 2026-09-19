# Phoenix private A/B: blinded free answers and human scoring (experimental)

**No real model training or 350M candidate has been executed by this work.** These scripts construct a blinded comparison and aggregate *human-provided*, frozen correctness decisions. Neither script creates a genuinely independent holdout, verifies source rights, checks semantic training contamination, calculates actual model answers, verifies self-declared generation settings against inference code or establishes any quality gain. Do not publish prompts, free answers, private source files, packet, grading notes or secret key.

## 1. Independently prepare and freeze inputs

An independently authored and reviewed UTF-8 JSON holdout is `{"schema":1,"cases":[...]}`, with 8–500 entries containing **exactly** `id`, `language` (`de`/`en`), `domain`, `prompt`, `rubric`. IDs and normalized prompts must be unique; both languages required. Public Neral development fixtures and the previously tested eight cases **cannot** be reused as an independent holdout. Independently compare against all training, validation and earlier development prompts for exact **and semantic** leakage, verify rights and privacy, then record the holdout file SHA-256 before running either model. The script does not implement or certify this audit.

Prepare two genuine local inference JSON files with exact top-level keys `schema`, `checkpoint_sha256`, `tokenizer_sha256`, `holdout_sha256`, `max_new_tokens`, `decode`, `cases`. The schema is integer `1`; digests are lowercase 64-character SHA-256 strings. Original model checkpoint must hash to `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`; candidate must hash differently. Both must identify the *identical real* tokenizer and exact holdout bytes, max tokens and full decoding parameters. Independently hash real files to verify all report claims. The scripts only validate what the JSON says.

Report each case in **exact holdout order** with keys `id`, `prompt_sha256`, `input_ids_sha256`, `answer`, `generated_tokens`, `finish_reason`. The prompt digest is SHA-256 of the exact UTF-8 prompt. Token-ID SHA must hash the same canonical serialization for both runs, including actual BOS/chat markers; e.g. `json.dumps(token_ids, separators=(',', ':'), ensure_ascii=False).encode('utf-8')`. Capture actual answer and count (<= cap), with `finish_reason` one of `eos`, `max_tokens`, `role_boundary`, `repetition_guard`. Both reports must have identical per-case token-ID hashes. Do not invent these metadata or treat a SHA field as proof that an inference run occurred.

## 2. Create blinded packet and secret key

Run only locally, after independently approved evaluation, with existing private folder ending in `NeralPrivateAssets/phoenix`:

```bash
python frontier350/nextgen_v1/phoenix_blind_ab_audit.py --execute --holdout PATH_TO_FROZEN_HOLDOUT.json --baseline PATH_TO_REAL_ORIGINAL_OUTPUT.json --candidate PATH_TO_REAL_CANDIDATE_OUTPUT.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01
```

Creates new files `phoenix_blind_pilot01.json` (human grader sees this, NOT the key) and `phoenix_blind_pilot01.key.json` (secret A/B assignment per case). Each case is independently shuffled A/B. Key commits to exact serialized packet SHA and both per-answer hashes so a key from a different run on the *same holdout* or altered answers is rejected. Files never overwrite existing names; keep both private. The scripts provide no cryptographic signature, protected key storage or guarantee of grader independence.

## 3. Human grades, freezes and only then unblinds

Human grader receives ONLY the packet and the predeclared rubric, not the secret key or checkpoint identifiers. Prepare a separately frozen JSON file with exactly:

```json
{"schema":1,"packet_sha256":"<SHA-256 of exact phoenix_blind_pilot01.json bytes>","items":[{"id":"case_1","A_correct":true,"B_correct":false,"notes":"Explain concrete rubric-grounded decision"}]}
```

The `items` array must contain **every** packet ID **once in the original order**, with exactly `id`, strict Boolean `A_correct`, `B_correct` and a nonempty human rationale `notes` (max 2000 chars). Do not use demonstration values above as real grades. Independently save the grading file SHA and verify the human grades are frozen before revealing the key. The software checks structural completeness and hashes; it cannot determine whether a human truly graded blind.

After freezing grades, locally run:

```bash
python frontier350/nextgen_v1/phoenix_blind_ab_unblind.py --execute --packet D:/NeralPrivateAssets/phoenix/phoenix_blind_pilot01.json --grades PATH_TO_FROZEN_GRADES.json --secret-key D:/NeralPrivateAssets/phoenix/phoenix_blind_pilot01.key.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01
```

This creates `phoenix_scored_pilot01.json` (create-only). Summary contains totals, matched pair counts `baseline_only`, `candidate_only`, `both_correct`, `both_wrong`, a net **count** difference, and domain/language splits. It prints no answer text or grading notes. `quality_improvement_proven` remains `false`: these counts are *human-reported*, not verified ground truth; a small pilot or positive net alone does not prove generalized improvement. Do not change scoring/rubrics after unblinding or select checkpoints on holdout scores.

## 4. Synthetic tests and deployment state

From `frontier350/nextgen_v1`:

```bash
python -m unittest -v test_phoenix_blind_ab_audit.py test_phoenix_blind_ab_unblind.py
```

18 local synthetic tests pass in the isolated test environment, including full private-file flow, no-overwrite, original SHA, mismatch rejection, fake/altered grading, different packet key and answer substitution. **GitHub hosted Actions remain unverified/failed**, so do not call hosted CI green; real owner-PC execution of the new scripts remains outstanding. Training still needs human source/holdout reviews, a separate actual authorized run, real VRAM/thermals/disk checks, full 350M checkpoint recovery and independently scored original/candidate free answers. PR #118 and this PR remain drafts; issue #119 remains open. Never upload private data or model tensors to public CI.
