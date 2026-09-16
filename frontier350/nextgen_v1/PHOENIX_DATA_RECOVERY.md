# Phoenix data recovery after both private inputs are lost

**Situation verified on 2026-09-16:** the self-hosted preflight found the configured private root but neither the pinned OASST2 archive nor the prior frozen holdout registry at its expected path. The operator subsequently confirmed both are unavailable. This is a **new evaluation campaign**, not restoration of the original benchmark. No data, weights, or private prompts have been uploaded; no download, training, or new benchmark creation is performed by this document.

## 1. Recover the pinned source archive only (manual, opt-in)

The exact upstream file exists at `OpenAssistant/oasst2`, revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`, under Apache-2.0 dataset-level metadata. Hugging Face lists the archive at 54,301,900 bytes and SHA-256 `a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751`. Upstream metadata does not replace row-level provenance/privacy/license review. Source: https://huggingface.co/datasets/OpenAssistant/oasst2/blob/main/2023-11-05_oasst2_ready.messages.jsonl.gz and pinned tree https://huggingface.co/datasets/OpenAssistant/oasst2/tree/179dd21fc55192153d94adb0e0ce8f69e222bf75.

Run these commands **on the owner's Windows PC, from PowerShell, only if a ~54 MB download is wanted**. They do not process corpus text or start training; they do not overwrite a previous archive and never upload raw data. Check `NERAL_ASSETS_ROOT` yourself; use a private folder excluded from git.

```powershell
$ErrorActionPreference = 'Stop'
$root = if ($env:NERAL_ASSETS_ROOT) { $env:NERAL_ASSETS_ROOT } else { Join-Path $env:USERPROFILE 'Documents\Codex\2026-09-05\lo\outputs\neral' }
if (!(Test-Path -LiteralPath $root -PathType Container)) { throw 'Private assets root not found; verify NERAL_ASSETS_ROOT first.' }
$source = Join-Path $root 'sft50_sources'
New-Item -ItemType Directory -Path $source -Force | Out-Null
$archive = Join-Path $source '2023-11-05_oasst2_ready.messages.jsonl.gz'
$temp = "$archive.partial"
if ((Test-Path -LiteralPath $archive) -or (Test-Path -LiteralPath $temp)) { throw 'Archive or partial already exists: inspect locally; no overwrite.' }
$url = 'https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/2023-11-05_oasst2_ready.messages.jsonl.gz'
$expected = 'a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751'
try {
    Invoke-WebRequest -Uri $url -OutFile $temp -UseBasicParsing
    $item = Get-Item -LiteralPath $temp
    if ($item.Length -ne 54301900) { throw 'Unexpected archive byte count.' }
    if ((Get-FileHash -LiteralPath $temp -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expected) { throw 'Pinned SHA-256 mismatch. Stop.' }
    Move-Item -LiteralPath $temp -Destination $archive -ErrorAction Stop
    Write-Host 'Verified pinned OASST2 archive exists locally. Dataset NOT approved; no training started.'
} catch {
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force }
    throw
}
```

If the upstream content cannot be fetched or its digest differs, stop. Never change the pinned digest to match a different download or replace the source with synthetic/scraped data. The earlier `fetch_oasst2_sft50.py` is a different, broader conversion pipeline; for Phoenix use `phoenix_oasst2_bridge.py` only **after** a properly frozen new holdout exists.

## 2. Establish a genuinely new benchmark, named v2

The original private `holdout_hashes.json` has been lost. **Do not fabricate the previous contents, re-label eight existing smoke prompts as an untouched benchmark, or declare historical NIS2 scores comparable to a new test.** Before inspecting or training a candidate model, author and review a new private prompt set (e.g. 24–100 unique, independently written, mix of DE/EN, factual reasoning, math, coding, multilingual dialogue and safety). Include IDs, task categories and a rubric/reference answer where objectively gradable; preserve raw questions/answers *only locally*. Avoid reusing source conversations, train/validation samples, any existing engineering smoke prompts, or published NIS2 answers.

Normalize prompt text with `sft_contract.prompt_hash()` (NFKC, Unicode casefold, whitespace normalization, SHA-256) and persist an ordered, nonempty list of **distinct** 64-character lower-case hashes as a new local `phoenix/holdout_hashes.json` **only after** verifying it represents the new v2 set. The path is the parser's legacy interface filename; the *contents and provenance are v2*, not a recovery of the original. Store a separate private immutable `phoenix/holdout_v2_receipt.json` with `campaign: phoenix-holdout-v2`, creation UTC time, question count, SHA-256 of the raw private question file, SHA-256 of the registry file, source/authorship and an explicit `not_comparable_to_prior_holdout: true`; preserve both and their backups outside git. Freeze the question set and receipt before producing SFT training/validation files. Do not create an empty or dummy registry solely to satisfy the file-existence check. The current preflight tests file existence/size, **not independent benchmark provenance or quality**.

Separate phases clearly: technical checksum, independent data-license/privacy/PII/semantic-overlap review, written approval, bounded SFT pilot with copied weights, matched A/B and v2 NIS2. Never promote solely on reduced validation loss. The adapter's `approved: false` must remain false; manual sign-off is recorded separately.

## 3. Verify and prepare only after v2 freeze

With **both** the verified archive and v2 holdout registry in the configured private root, run the read-only preflight (`.github/workflows/phoenix-oasst2-preflight.yml` is automatically triggered only on modifications to itself). Then follow `PHOENIX_OASST2.md` to export a new local candidate subset using a *new output path*, review its composition and screen it with `phoenix_data_audit.py`. If no independent holdout can be frozen, do not start SFT. The prior eight-prompt 64-token smoke remains an engineering diagnostic, not a newly independent quality score.
