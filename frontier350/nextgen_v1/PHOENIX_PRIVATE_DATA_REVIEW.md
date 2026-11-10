# Phoenix OASST2: private human review (experimental; NOT approved for training)

## Verified local state, 2026-09-17

The original SHA-pinned OASST2 source and all human-review files are stored **outside Git** under `D:\NeralPrivateAssets`. No source dialogue, review notes or identifiers were printed or uploaded to GitHub.

- The screened experimental selection is **1,200 candidates**; the pending review queue samples **24 original examples** with six examples per DE/EN × train/validation stratum. The other 1,176 candidates have not been manually reviewed. One repeated normalized assistant answer in the 1,200-candidate inventory still needs investigation.
- Original immutable queue: `D:\NeralPrivateAssets\phoenix\screened_review_queue_v1.jsonl`.
- Original queue receipt: `D:\NeralPrivateAssets\phoenix\screened_review_queue_v1_receipt.json`.
- Queue SHA-256: `94053e24e34b309ced74cf0ecdf75db484b4168718a260039ae3b7931c579f81`.
- **NEW recommended offline page, created and confirmed:** `D:\NeralPrivateAssets\phoenix\screened_review_queue_v2.html`. Double-click this local file. The old `screened_review_queue_v1.html` was not overwritten and can remain as an archival fallback.
- Evidence: [original local queue #35204636255](https://github.com/Cola968/neral-helios/actions/runs/35204636255), [local v1 page #35204793825](https://github.com/Cola968/neral-helios/actions/runs/35204793825), [local v2 page #35216647249](https://github.com/Cola968/neral-helios/actions/runs/35216647249). The v2 run independently verified the unchanged original queue SHA, created a new 67,168-byte HTML file and did not overwrite v1 or use a GPU.

The [automated 24-case read-only check #35206329625](https://github.com/Cola968/neral-helios/actions/runs/35206329625) found **zero warning flags in this sample under its particular regex/length/duplicate heuristics**, not zero possible errors. It did not judge factual correctness, license, safety, source provenance or semantic test overlap.

## Human reviewer: use the offline v2 page

For each of 24 real dialogues, read both prompt and answer and record **OK or Ablehnen for four separate checks**: privacy/PII, attribution/provenance, answer quality and semantic overlap with an *independently authored, frozen* holdout. There are 96 individual decisions. If the necessary source, benchmark or evidence is absent, choose Ablehnen rather than treating the missing check as complete; use a brief private note.

**You can now stop and resume without browser storage:** click **Zwischenstand lokal als JSON sichern**, retain the downloaded `phoenix_review_draft_v2.json` only on your own PC, and later use **Zwischenstand vom Computer laden** in the same v2 page. The page validates the entire file's queue SHA, row IDs, schema and choice values before replacing your current entries and asks confirmation if it would overwrite unsaved inputs. It does *not* auto-save in cookies or localStorage. Secure the draft before closing the page. An incomplete draft is **not** accepted as a finished human review by `phoenix_review_decisions.py`.

Once all 96 choices are set, the final button becomes available and exports `phoenix_review_decisions_v1.json`, using the existing independent decision-validator schema. The browser normally places drafts/final exports in Downloads; both contain private identifiers and possibly notes. **Move only the final JSON** to `D:\NeralPrivateAssets\phoenix\phoenix_review_decisions_v1.json` for validation. Keep drafts private and do not upload either to GitHub, attach them to chat or commit them. Nothing auto-deletes files from Downloads; remove surplus copies manually after verifying your private backup.

## Validate final export without approving training

`phoenix_review_decisions.py` compares the final JSON with the exact SHA-pinned original queue and receipt, checks all 24 exact IDs and four valid choices per row, bounded notes, duplicate JSON keys, path isolation and output create-only semantics. It rejects `training_approved: true`, changed SHA, drafts with missing choices, foreign IDs and missing rows. Its output is an aggregate SHA-bound **non-approval receipt only**. Neither the UI nor validator can verify reviewer identity or correctness of subjective decisions.

Manual-only CPU workflow `.github/workflows/phoenix-review-decisions-manual.yml` should be run only *after* the final exported file is in the exact path above; workflow dispatch may not appear in GitHub until the workflow exists on the repository default branch. Do **not** merge the whole experimental PR just to expose a button. Alternatively, from an up-to-date checkout of `experiment/phoenix-sft-fp32-audit` use the local WSL interpreter:

```bash
~/.venvs/neral-triton-py312/bin/python frontier350/nextgen_v1/phoenix_review_decisions.py \
  --assets-root /mnt/d/NeralPrivateAssets \
  --queue /mnt/d/NeralPrivateAssets/phoenix/screened_review_queue_v1.jsonl \
  --receipt /mnt/d/NeralPrivateAssets/phoenix/screened_review_queue_v1_receipt.json \
  --decisions /mnt/d/NeralPrivateAssets/phoenix/phoenix_review_decisions_v1.json \
  --out /mnt/d/NeralPrivateAssets/phoenix/phoenix_review_decisions_v1_validated_receipt.json \
  --queue-sha256 94053e24e34b309ced74cf0ecdf75db484b4168718a260039ae3b7931c579f81
```

The receipt path must not already exist; no original queue, v1/v2 HTML or decisions file is overwritten. [Tested v2 UI, JavaScript syntax and source integrity](https://github.com/Cola968/neral-helios/actions/runs/35216572022).

## STOP/GO remains CLOSED

Independent 24–100-question DE/EN holdout v2 with reviewed rubrics is **not yet frozen**. Semantic overlap, source-license/attribution and human review of a much larger representative candidate population plus independently signed SHA-bound approval remain missing. A passing automated 24-case audit and even 96 later manual decisions cannot certify all 1,200 rows. No training has run; no new model quality, NIS2 score or 60–70k accepted tokens/s claim can be made. PR #118 stays draft/unmerged and SAFE weights stay unchanged.
