# Helios guarded 350M chat — experimental local preview (17 September 2026)

**Status: tested engineering preview, NOT a new model and NOT approved chat quality.** Code lives only in the draft Phoenix branch. No modified model weights, SFT, GPU execution, network upload, checkpoint copies or train/validation approval.

## Current abilities and boundaries

- The own 350M-parameter model has a verified continuity checkpoint after **2,073,161,216 pretraining tokens**; it generates text. The old 1.010B SAFE model is a separate checkpoint, not the current continuation.
- The original `helios350_preview.py` is a local single-turn browser interface. The new `helios_guarded_preview.py` is an **optional separate CPU-only single-turn mode** on port `9682`; the original raw greedy decoder and evaluation protocols are untouched.
- This mode applies a 1.18 penalty to tokens already emitted in the last 32 output tokens, stops before a visible new `User:`, `Assistant:`, `System:`, `Nutzer:` or `Benutzer:` role line, and stops when a narrow word-trigram-loop heuristic triggers. It also respects native EOS or the 64-token ceiling. The API JSON returns the true `finish_reason` (`eos`, `max_tokens`, `role_boundary`, `repetition_guard`); the HTML currently displays answer text only.
- Input: max 1,024 characters and 384 token IDs; output: max 64 new token decisions; server only binds to `127.0.0.1`. No chat history, tools, actual reasoning assurance, correctness guarantee or benchmark validity. CPU inference can be slow.

## Verified real hardware evidence

[Workflow #35231390557](https://github.com/Cola968/neral-helios/actions/runs/35231390557): **16/16 synthetic tests pass** and a real one-shot CPU diagnostic on the exact SHA-pinned 2.073B checkpoint completed. Eight *public engineering* prompts, budget 32: 8/8 nonempty visible responses; 5 native EOS, 2 role-boundary stops, 1 token-budget stop, 0 repetition-guard triggers; visible role bleed 0 and repeated word-trigram ratio 0 **for this small guarded sample only**; 89 tokens counted including native EOS, generation phase 16.49 seconds. No answers were uploaded or printed. This is not a matched raw-vs-guarded correctness trial; the policy changes decoding, so raw Phoenix A/B evidence must retain its original protocol. `correctness_evaluated=false` and `independent_quality_benchmark=false`.

## Start locally, only if you want an interactive session

1. Open the experiment checkout on your PC. The successful workflow reused the source folder `C:\actions-runner\_work\neral-helios\neral-helios\phoenix_guarded_preview_source\frontier350\nextgen_v1`. If you use a different checkout, update that branch there first.
2. Double-click `START_HELIOS_GUARDED.cmd` in that folder. It launches its companion `start_helios_guarded.ps1` with a temporary process-scoped PowerShell execution-policy override; it does **not** change system policy or install anything. The launcher verifies free-space reserves and SHA-256 of the existing D: checkpoint, then starts the CPU-only WSL server. An explicitly configured `NERAL_ASSETS_ROOT` takes precedence; otherwise it uses the documented existing user-profile asset location.
3. Wait for the terminal message `Open http://127.0.0.1:9682/`, then open **http://127.0.0.1:9682/** in a browser on the *same* PC. Stop with Ctrl+C in the terminal. The URL is not a remotely hosted website and won't work when the local server is stopped.

The UI returns only the answer; the `/generate` JSON additionally includes `finish_reason`. No prompts or responses are saved by the preview. The model and tokenizer are loaded from existing locations, never duplicated on C: or D:.

## Unresolved, and must not be confused with this delivery

A genuine independently prepared and frozen holdout v2; source license/privacy/content review; a measured correctness baseline; a bounded authorized SFT pilot; identical-protocol post-training A/B; robust multi-turn context. The 1,192-example triaged candidate remains **unapproved**, and the eight quarantined records must not be automatically presumed wrong. PR #118 remains draft/unmerged; production weights unchanged.
