# Phoenix: pinned OASST2 offline source bridge

**Status: implemented local converter; not an approved dataset and not a training run.** The project already contains a pinned human-only OpenAssistant-2 selector (`fetch_oasst2_sft50.py`) and its source policy. Phoenix reuses the selector but **does not** reuse an old SFT split or claim that a prior model's approval transfers to a new campaign.

## What the converter actually verifies

`phoenix_oasst2_bridge.py` accepts ONLY a locally present, exact-SHA-256-matched, pinned OpenAssistant-2 ready-message archive at revision `179dd21fc55192153d94adb0e0ce8f69e222bf75` (archive SHA-256 in `fetch_oasst2_sft50.py`). No downloads or GitHub uploads occur. The older selector keeps positively reviewed, nondeleted, nonsynthetic DE/EN prompter/assistant pairs. The bridge independently verifies their raw message IDs, parent linkage, matching prompt/answer text, human review, language and shared `message_tree_id`.

Every complete conversation tree receives ONE deterministic split, including German/English turns in the same tree. The exporter removes duplicate normalized prompts and incompatible marker/length rows, and rejects any exact collision with a separately frozen private holdout-hash registry. It stops if the archive lacks consistent tree IDs or either split is too small. A report contains only counts, hashes, split method and `approved: false`. Training/validation JSONL contain the four fields required by `sft_contract.py`: `id`, `source_id`, `user`, `assistant`.

**Important limitations:** this is a deterministic, order-dependent capped *development subset*, not a random population sample. The simple 90/10 tree hash does not guarantee a balanced domain/language distribution. It does not detect semantic overlap, every near duplicate or every item of personal information. The SHA protects file identity, not legal or factual quality. A human must review representative source examples, licensing/attribution, privacy, DE/EN/domain composition, contamination and the selection receipt before issuing an independently documented approval. The code NEVER sets an approval flag to true.

## Local operator commands, only using existing private files

PowerShell from the GitHub checkout. The environment variable `NERAL_ASSETS_ROOT` must identify an existing private root containing BOTH the original OASST2 archive and the frozen holdout-hash file. The paths below are illustrative: **verify actual locations**; do not fabricate a holdout or run a download automatically.

```powershell
$root = $env:NERAL_ASSETS_ROOT
$archive = Join-Path $root 'sft50_sources/2023-11-05_oasst2_ready.messages.jsonl.gz'
$holdout = Join-Path $root 'phoenix/holdout_hashes.json'
$out = Join-Path $root 'phoenix/oasst2-source-new'
python frontier350/nextgen_v1/phoenix_oasst2_bridge.py --assets-root "$root" --archive "$archive" --holdout-hashes "$holdout" --out "$out" --max-en 800 --max-de 400
```

If the archive, the original `message_tree_id`, the hash registry or the output parent is unavailable, **stop**; do not silently reconstruct identities from a message-level split or download another revision. The output directory must be new, under the private root. Original source text and all generated row files stay local and must never be committed to GitHub.

## Next gates, still unfulfilled

1. Inspect `source_report.json` locally. Review the selected source and every excluded/flagged category; independently verify the actual dataset terms, attribution, source revision, private information and test contamination. Source policy alone is not approval.
2. Run `phoenix_data_audit.py` against the two locally produced JSONL files and the SAME holdout registry. Its own `approved: false` diagnostic is not an independent clearance.
3. Only after external/human review, create a new, separately documented screen report and SHA-pinned source manifest as described in `SFT_PILOT.md` and feed that to `sft_contract.py`. NEVER flip the bridge/prescreen `approved` flag to bypass this requirement.
4. Once the data are truly approved, check for a stable own-weight checkpoint copy and an idle RTX 4070, then consider the bounded 1–50-step isolated pilot. Its lower validation loss, if any, is NOT evidence of better chat responses: use independent matched baseline/candidate generation and NIS2 afterward.

No actual full OASST2 archive was loaded or vetted in GitHub CI. The bridge tests use a synthetic miniature gzip fixture and a *test-only* SHA override. The production entry point always enforces the existing repository-pinned hash. No new weights, model quality or GPU throughput are claimed by this document.
