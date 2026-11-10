# Phoenix Holdout v2: private rubric binding (NO automatic benchmark approval)

The original private holdout is missing. Historic NIS2 scores cannot be compared with new-v2 scores. Do **not** fabricate old prompts or use the eight public engineering smoke prompts as the holdout. This is a new campaign only.

## Required private inputs

Under a private assets root outside Git, independently author and review 24–100 new prompts with distinct IDs, including German and English, and freeze them as described in `PHOENIX_HOLDOUT_V2.md`. The freezing script creates `holdout_hashes.json` and `holdout_v2_receipt.json` with immutable content digests; it does not establish authorship or fairness.

Create a **separate private** JSON array `holdout_rubrics_v2.json`, with one scoring rubric per question ID. Each entry has exactly `id`, `language`, `category`, `pass_condition`. For each rubric, choose `language` = `de` or `en`, a category from `factual`, `reasoning`, `math`, `instruction`, `safety`, `writing`, `code`, and write a 20–1000-character concrete acceptance rule. The `id` MUST equal the corresponding private prompt's ID. The format below is demonstrative only and is not a real benchmark:

```json
[
  {
    "id": "EXAMPLE-ONLY",
    "language": "en",
    "category": "factual",
    "pass_condition": "Specify an independently verifiable acceptance rule for this question; this example must not be scored."
  }
]
```

The example is intentionally insufficient to pass the 24-question minimum. The checker requires at least 4 questions in each language, 3 task categories, exact one-to-one rubric/question ID coverage, and the original freeze digests. Design the rest of the language/skill mix before looking at any candidate model outputs. Store the raw prompts, rubrics, receipts and hashes privately; never upload them to GitHub, Actions logs or connected apps.

## Read-only shape and freeze verification

In a trusted Python environment from the checked-out repository root, use an existing private root explicitly; do not install new Windows Python or modify registry/policies:

```powershell
$root = $env:NERAL_ASSETS_ROOT
if (!$root -or !(Test-Path -LiteralPath $root -PathType Container)) { throw 'Existing private assets root required' }
python frontier350/nextgen_v1/phoenix_holdout_rubric_check.py --assets-root "$root" --prompts "$root/phoenix/new_private_prompts_v2.json" --registry "$root/phoenix/holdout_hashes.json" --receipt "$root/phoenix/holdout_v2_receipt.json" --rubrics "$root/phoenix/holdout_rubrics_v2.json"
```

A successful check prints counts, task/language distribution and hashes **only**. If the private v2 files have not been created yet, the checker must fail. It does not create questions, score answers, assess rubric quality, check semantic overlap, verify authorship, review privacy/license, export training rows, approve data, launch CUDA or train a model.

## Conditions not automated here

Before any pilot training, separately have an independent reviewer check prompt authorship, rubric answer correctness, privacy, contamination (exact and semantic) against both candidate training data and legacy benchmark items, and that the planned scoring rule can be applied blind to both SAFE and candidate. Preserve the new population and evaluation protocol unchanged before testing candidates. A passing structural check is **necessary but not sufficient** for a defensible quality claim, and cannot be used to change `approved:false` on any OASST2 report.
