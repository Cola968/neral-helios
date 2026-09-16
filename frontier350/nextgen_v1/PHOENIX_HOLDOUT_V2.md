# Phoenix: create a NEW private evaluation holdout (v2)

The old holdout is missing, so **never claim these are restored original NIS2 questions or comparable scores**. This tool freezes only the user's freshly curated questions. No invented questions, downloads, candidate training, scoring, licence/PII approval, or file uploads.

1. From the private assets root, create `phoenix/new_private_prompts_v2.json` **yourself**, with 24–100 genuinely new, independently reviewed questions. JSON format: an array of objects each with exactly `id` (unique ASCII letters/digits/`_`/`-`) and `prompt` (20–2000 characters). Example of the *format only* below; **two example questions are not sufficient**. Write a balanced DE/EN mix with objective and qualitative rubrics in a separate locally preserved file. Do not reuse eight existing engineering smoke prompts, source conversation rows or old benchmark answers.

```json
[
  {"id":"new-de-001","prompt":"Eine eigens formulierte Frage mit ausreichend Text hier eintragen."},
  {"id":"new-en-002","prompt":"Write another independently authored question with enough context here."}
]
```

2. From the checked-out repository root, run PowerShell, using an existing private `NERAL_ASSETS_ROOT`. Confirm the actual root is excluded from git before invoking:

```powershell
$root = $env:NERAL_ASSETS_ROOT
if (!$root -or !(Test-Path -LiteralPath $root -PathType Container)) { throw 'Configure an existing private NERAL_ASSETS_ROOT first.' }
python frontier350/nextgen_v1/phoenix_freeze_holdout_v2.py --assets-root "$root" --prompts "$root/phoenix/new_private_prompts_v2.json" --registry "$root/phoenix/holdout_hashes.json" --receipt "$root/phoenix/holdout_v2_receipt.json"
```

The path `holdout_hashes.json` is the **existing parser interface filename** but must now contain genuinely new v2 hashes, not invented legacy hashes. The receipt explicitly records `campaign=phoenix-holdout-v2`, both file digests and `not_comparable_to_prior_holdout=true`. Both outputs are new-file only: existing files are never overwritten. The program rejects duplicated normalized prompts and IDs, invalid schema, too few questions, and output paths outside the private root. It prints only aggregate status/count. It does NOT establish authorship or benchmark independence; manually verify rubric, PII and semantic source overlap. Preserve the raw questions, scoring rubric, hash list and receipt in a secure backup outside the repo; freeze before candidate training.

3. After recovering the exact pinned OASST2 archive from `PHOENIX_DATA_RECOVERY.md`, and after checking this new holdout for independence, run the existing local `phoenix_oasst2_bridge.py` as documented in `PHOENIX_OASST2.md`. Its outputs still say `approved: false`. Independently review selected training rows, legal terms, attribution, privacy, near-duplicates and semantic contamination. Only after separate approval should an isolated pilot be considered.

**Evidence boundary:** Passing `test_phoenix_freeze_holdout_v2.py` on synthetic examples means the file/format safeguards work; it cannot prove new questions are valid, that the real archive has been downloaded, or that model quality improved.
