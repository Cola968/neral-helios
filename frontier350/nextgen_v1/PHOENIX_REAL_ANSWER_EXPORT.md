# Phoenix: actual own-checkpoint free-answer export (experimental)

**Implemented but not yet executed with a genuine independent holdout or actual new candidate.** `phoenix_real_answer_export.py` connects the pinned own Neral 350M CPU loader to private, blind-A/B-compatible JSON reports. It does not train, authorize training, create a holdout, grade answers or prove a quality gain. Never commit the holdout, raw answers, grading file or A/B mapping key.

## Preconditions

1. Independently author and freeze an unseen bilingual DE/EN holdout in the schema documented in `PHOENIX_BLIND_AB_README.md`. The old public development questions are not independent. Human content/rights/privacy and exact **and semantic** contamination checks are required externally; the scripts cannot certify them.
2. Use the exact pinned original 350M checkpoint SHA-256 `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7` and original tokenizer SHA-256 `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`, with the asset root containing `frontier350_assets/tokenizer.json`.
3. A candidate must actually exist from the separately approved, bounded **weight-only/new-optimizer** pilot with its own full SHA-256, immutable source and independent output. There is no verified new 350M candidate today. A self-declared checkpoint metadata field or hash is not independent proof of approval or training.
4. Use an existing private `D:/NeralPrivateAssets/phoenix` directory and force CPU isolation before running (PowerShell: `$env:CUDA_VISIBLE_DEVICES='-1'`). Never execute the private holdout or true model evaluation in hosted CI.

## Private commands, only after prerequisites are met

Original:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python frontier350/nextgen_v1/phoenix_real_answer_export.py --execute --trusted-own-checkpoint --role original --assets-root D:/YOUR_REAL_NERAL_ASSET_ROOT --checkpoint D:/YOUR_PINNED_ORIGINAL/resume.pt --holdout D:/NeralPrivateAssets/phoenix/independent_frozen_holdout.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01 --max-new-tokens 64
```

Separately approved candidate, replacing the SHA placeholder with the **independently computed digest of the real checkpoint bytes**:

```powershell
python frontier350/nextgen_v1/phoenix_real_answer_export.py --execute --trusted-own-checkpoint --role candidate --assets-root D:/YOUR_REAL_NERAL_ASSET_ROOT --checkpoint D:/YOUR_ACTUAL_APPROVED_PILOT/resume.pt --candidate-sha256 YOUR_INDEPENDENTLY_RECORDED_FULL_SHA256 --holdout D:/NeralPrivateAssets/phoenix/independent_frozen_holdout.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01 --max-new-tokens 64
```

The candidate SHA is verified **before Torch deserialization**. The default uses `weights_only=True`; only for the operator's personally trusted, independently hash-verified **own** legacy checkpoint may the explicit `--allow-trusted-legacy-pickle` flag enable an unsafe legacy load. Never do so for untrusted weights.

`phoenix_candidate_lineage.validate_candidate` now additionally demands checkpoint schema v2, original source SHA, historical source token count separate from new zero-based counters, valid bounded run settings, full approval-hash metadata, matching **exact frozen holdout SHA**, correct own pilot `resume.pt` directory, target and step/token-count arithmetic, tokenizer, window sampler, train/validation identities, nonempty optimizer state and model dictionary. A mismatch raises before model construction. This is a **consistency check of self-reported fields**, not a cryptographic signature or a substitute for independently checking approval/source rights. Actual tensor/moment correctness and GPU feasibility are not certified by metadata inspection.

The exporter checks original tokenizer round-trip, forms `BOS + tokenize('User: ' + prompt + '\nAssistant:')`, preserves the **input ID hash before generation appends tokens**, and generates up to the identical 64-token cap via CPU float32 greedy argmax. EOS counts as a generation step. Outputs are create-only `helios_quality_ab_original_pilot01.json` and `helios_quality_ab_candidate_pilot01.json` inside the private directory. Compare these with `phoenix_blind_ab_audit.py`, independently freeze human grades and use `phoenix_paired_evidence.py` only for exploratory paired statistics.

## Actual evidence on 19 September 2026

Five synthetic exporter tests and eight synthetic candidate-lineage tests passed locally; all 50 local Phoenix evaluation/data-audit tests together passed. This does **not** mean actual 350M checkpoints were loaded in the new exporter: synthetic stubs and fabricated checkpoint metadata were used. Hosted GitHub CI remains failing before runner assignment/steps with no retrievable logs; its root cause has not been verified. The original eight public free-answer development cases were 0/8 strict; there is no newly trained model, independent A/B measurement, proven quality gain or measured training throughput. Keep both PRs draft and issue #119 open.
