# Phoenix: actual own-checkpoint free-answer export (experimental)

**Implemented generator, not an executed 350M evaluation or trained candidate.** `phoenix_real_answer_export.py` connects the existing SHA-pinned own Neral 350M CPU model loader and greedy decoder to the private A/B report schema. It executes no training, creates no checkpoint or holdout, and neither scores responses nor claims independence or improvement. It keeps the frozen holdout and ALL generated answers exclusively on the owner's private machine, never in logs, CI, or GitHub.

## Prerequisites

- A genuinely independent, separately human-authored, bilingual, frozen holdout (`{"schema":1,"cases":[...]}`) with all five documented fields; manually check licensing/privacy/semantic contamination separately. Public Neral development questions do **not** qualify.
- Original 350M snapshot, exactly SHA-256 `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`, original tokenizer hash `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd` and `frontier350_assets/tokenizer.json` in the local asset root.
- A distinct, actually existing and independently SHA-pinned **bounded weight-only-pilot** checkpoint is required for the candidate run. It must carry matching Neral architecture/tokenizer and `explicit_weight_only_new_optimizer_v1` lineage originating from the original 350M checkpoint; it cannot be fabricated by editing report JSON. None is reported to exist yet. The candidate must be reviewed and produced under the separate authorization requirements.
- Existing private `D:/NeralPrivateAssets/phoenix` directory. Run locally with `CUDA_VISIBLE_DEVICES=-1` set before launch. On PowerShell, `$env:CUDA_VISIBLE_DEVICES='-1'`. Do not run private holdout inference in hosted CI.

## Exact commands (only when real files and approvals exist)

Run the baseline locally, substituting absolute **real** paths:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python frontier350/nextgen_v1/phoenix_real_answer_export.py --execute --trusted-own-checkpoint --role original --assets-root D:/YOUR_REAL_NERAL_ASSET_ROOT --checkpoint D:/YOUR_PINNED_ORIGINAL/resume.pt --holdout D:/NeralPrivateAssets/phoenix/independent_frozen_holdout.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01 --max-new-tokens 64
```

Run only after a separately approved pilot has actually produced a new distinct checkpoint, replacing the placeholder full digest:

```powershell
python frontier350/nextgen_v1/phoenix_real_answer_export.py --execute --trusted-own-checkpoint --role candidate --assets-root D:/YOUR_REAL_NERAL_ASSET_ROOT --checkpoint D:/YOUR_ACTUAL_APPROVED_PILOT/resume.pt --candidate-sha256 YOUR_INDEPENDENTLY_RECORDED_FULL_64_CHARACTER_SHA256 --holdout D:/NeralPrivateAssets/phoenix/independent_frozen_holdout.json --private-dir D:/NeralPrivateAssets/phoenix --tag pilot01 --max-new-tokens 64
```

Candidate deserialization uses PyTorch `weights_only=True` by default. If and only if that fails on the operator's **personally trusted, exactly hash-verified own** legacy checkpoint, a deliberate `--allow-trusted-legacy-pickle` opt-in permits loading its legacy pickle. Never enable it for files from others; hashes alone do not make arbitrary pickle safe.

The exporter uses the exact previously audited preview prompt tokenizer `BOS + tokenize('User: ' + prompt + '\nAssistant:')`, strict round-trip, full CPU float32 greedy argmax and identical 64-token ceiling. EOS is counted as a generation step. Crucially, it hashes the **unchanged original input IDs before generation** (decoders append new IDs), not the already expanded input. It verifies absolute candidate path and actual SHA before any potentially unsafe Torch load; candidate config/tokenizer/lineage/step bounds are then checked. It refuses wrong roots, output collisions, missing inputs and unsupported sequences.

Private outputs are `helios_quality_ab_original_pilot01.json` and `helios_quality_ab_candidate_pilot01.json` in the private directory. These directly match the generation schema consumed by `phoenix_blind_ab_audit.py`. Once both exist, use `PHOENIX_BLIND_AB_README.md` to create a blinded packet and obtain **independent frozen human grades**, then `PHOENIX_PAIRED_EVIDENCE.md` for an exploratory paired audit. Do not print or publish the response files, the secret mapping, rubrics or grading notes.

## Evidence boundary

Five new synthetic contract tests passed locally: integration with A/B schema, protection against input digest corruption by autoregressive append, invalid token/stop reports, context/cap validation and hash failure **before Torch import**. Full local A/B + evidence + exporter suite: 31 synthetic tests passed. No actual original or candidate checkpoint was loaded as part of these tests, no actual holdout exists here, and no quality or speed gains have been measured. A separately real, read-only CPU inference run on the owner's PC is still needed once reviewed holdout and candidate exist. Keep parent PR #118 and PR #120 draft; training remains unapproved.
