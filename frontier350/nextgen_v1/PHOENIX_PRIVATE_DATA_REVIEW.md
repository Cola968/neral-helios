# Phoenix OASST2: private human review (experimental, NOT approved for training)

## Verified local state, 2026-09-17

- The SHA-pinned archive was copied once into `D:\NeralPrivateAssets\sources\` OUTSIDE any Git working tree. No source dialogue is committed or uploaded.
- The sample has **24 real OASST2 rows**, exactly six in each DE/EN × train/validation stratum. It is derived from 1,200 screened candidates, but is **not** a certification of the other 1,176 rows. The source inventory has one repeated normalized assistant answer; more screening and human review are needed.
- Original private queue: `D:\NeralPrivateAssets\phoenix\screened_review_queue_v1.jsonl`.
- Original non-approval receipt: `D:\NeralPrivateAssets\phoenix\screened_review_queue_v1_receipt.json`.
- Original queue SHA-256: `94053e24e34b309ced74cf0ecdf75db484b4168718a260039ae3b7931c579f81`.
- **Open this already generated local HTML by double-clicking:** `D:\NeralPrivateAssets\phoenix\screened_review_queue_v1.html`.
- Generation evidence: [private 24-row queue](https://github.com/Cola968/neral-helios/actions/runs/35204636255) and [local offline HTML](https://github.com/Cola968/neral-helios/actions/runs/35204793825). Logs show aggregates only; no text, IDs, model or GPU access.

## Human operator: what to do in the offline page

For each of 24 dialogs, select OK or Ablehnen for all four fields: privacy/PII, attribution/provenance, response quality and semantic overlap with your **independently authored holdout**. Inspect actual answers, license and safety; if unsure select Ablehnen and add a brief private note. There are 96 decisions. The current v1 page exports only after all 96 have been chosen; it does NOT automatically save progress. Do not close the page mid-review without completing it. The export is named `phoenix_review_decisions_v1.json`; your browser normally places it in Downloads.

**Move** this JSON file locally to `D:\NeralPrivateAssets\phoenix\phoenix_review_decisions_v1.json`. The export can contain your private notes and dataset IDs: do not send it to GitHub, attach it to chat, upload it as an Actions artifact or commit it. After confirming the private copy, consider removing any extra copy in Downloads yourself; no script auto-deletes files.

## Validate export without approving training

`phoenix_review_decisions.py` compares the exported JSON against the exact SHA-pinned original queue and receipt, checks the 24 exact IDs, all four allowed choices per row, bounded private notes, duplicate keys, path isolation and no overwrite. It refuses an injected `training_approved: true`, altered queue hash, missing cases or foreign IDs. It writes only an aggregate, SHA-bound **non-approval** receipt into a NEW private file. It cannot prove who performed the review and does not judge the correctness of the reviewer's choices. The import tests use synthetic fixtures only: [CI](https://github.com/Cola968/neral-helios/actions/runs/35205676771).

A manual-only CPU workflow is staged as `.github/workflows/phoenix-review-decisions-manual.yml`. If GitHub exposes its **Run workflow** action on the experimental branch, select `experiment/phoenix-sft-fp32-audit` and run only AFTER the exported decisions file has been moved to the path above. GitHub may require a workflow-dispatch file to be present on the default branch before showing it; do NOT merge the entire experimental PR just to expose the button. The validator can alternatively be invoked locally using WSL and the experimental-branch source:

```bash
~/.venvs/neral-triton-py312/bin/python frontier350/nextgen_v1/phoenix_review_decisions.py \
  --assets-root /mnt/d/NeralPrivateAssets \
  --queue /mnt/d/NeralPrivateAssets/phoenix/screened_review_queue_v1.jsonl \
  --receipt /mnt/d/NeralPrivateAssets/phoenix/screened_review_queue_v1_receipt.json \
  --decisions /mnt/d/NeralPrivateAssets/phoenix/phoenix_review_decisions_v1.json \
  --out /mnt/d/NeralPrivateAssets/phoenix/phoenix_review_decisions_v1_validated_receipt.json \
  --queue-sha256 94053e24e34b309ced74cf0ecdf75db484b4168718a260039ae3b7931c579f81
```

Run the command from a checkout of `experiment/phoenix-sft-fp32-audit`, not from an outdated checkout; or pass an absolute path to the script. The output is create-only; if the receipt exists it refuses to overwrite it. Do not edit original queue or receipt.

## Gates that are STILL CLOSED

A manually curated, frozen, independent holdout v2 with 24–100 reviewed private questions and documented rubrics; semantic overlap screening; legal/source approval; broader stratified sample review; independent, SHA-bound training authorization; and a real reproducible before/after correctness evaluation are all still missing. Even if all 24 sample answers receive OK, **`training_approved` remains false**. Neither this review tool nor its UI starts SFT, modifies model weights, allocates GPU or establishes improved answer quality.
