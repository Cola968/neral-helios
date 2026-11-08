# NERAL six-arm paired failure triage (read-only CPU)

Original evidence: successful [six-arm GPU run #42470178641](https://github.com/Cola968/neral-helios/actions/runs/42470178641) on 8 November 2026.

## Actual observed scores on the already consumed 72-case developer suite

| Model | 0-shot | 3-shot |
|---|---:|---:|
| Qwen2.5-Coder-7B-Instruct | 19/72 | 19/72 |
| NERAL v5 | 16/72 | 17/72 |
| NERAL v6.1 | 16/72 | 17/72 |

No v5→v6.1 new correct or lost correct cases under either shot condition (0/0 discordances). The small Qwen advantage is entirely from Qwen-only exact correct answers on the same prompt policy: 3/72 zero-shot and 2/72 three-shot. In maths all three systems scored 7/24 per condition. Zero-shot gaps: code trace 6 vs 4, reasoning 6 vs 5. Three-shot gaps: code trace 6 vs 5, reasoning 6 vs 5. These are **not** independent release results.

## What the new audit adds

`scripts/neral_six_arm_failure_triage.py` reviews the **saved 72×6 original text outputs** from that job locally, without any model calls, GPU training, internet fetch, user-code execution, or rewriting scores.

It first runs the pinned `checked_private_run()` to validate the generator, model IDs, source response SHA hashes and case order, then independently regenerates the locked six-arm scores. It fails closed if any scalar in the original report changed.

For each prompt policy (0 or 3 examples), it produces only aggregated counters:
- Qwen-only / adapter-only correct answers and their mathematics, reasoning and Python-code domains and source families.
- Proven incorrect final integers versus ambiguous, ungradable full answers versus incomplete replies. These categories are **not** interchangeable.
- v5-v6.1 matched correctness *and* distinct raw texts and distinct parsed final values; identical accuracy does not imply identical behavior.
- Per-family baseline/adapter success rates and common failure rates.

A **separate private** `private-cases.jsonl` contains prompts, expected answers and six original raw completions, ordered with Qwen-only losses first for human root-cause review. Only `scalar.json` is copied to the GitHub Actions upload path. No case IDs, prompts, private answers, code snippets, or completion texts appear in the public scalar artifact. All source file SHA-256 hashes are included.

## Manual read-only analysis (no GPU model initialization)

Run `.github/workflows/neral-sixarm-private-triage-cpu-manual.yml` on `main` with

`authorize=AUDIT_NERAL_6ARM_PRIVATE_FAILURES_CPU`

The job checks the previously completed private source directory
`NERAL-QLoRA-Research\\sixarm-parity-42470178641-1` and creates two **new**
local files outside the repository, without modifying or uploading the
original model outputs. Its private review file remains on the Windows
machine. Public artifact has scalar counters only, with 14-day retention.

## Hard boundary

Use these already-consumed cases only to learn *what failed*. Do not select or train a new adapter using the 72 answers and then call gains on that same data a breakthrough. A genuine neural ability improvement needs a new, unreleased, independently constructed test, prior registered pass/fail criteria, and same-prompt Qwen/v5/candidate comparison. No checkpoint is promoted by this audit.
