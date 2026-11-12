# NERAL 1.1 – Python logic exploration (opt-in pilot, 2026-11-12)

## Scope

The normal LOW/HIGH endpoints and their raw Qwen baseline are unchanged.
This pilot is active only for an HTTP THINK request with **all three** boolean
flags `experimental_tir=true`, `conservative_tir=true`, and
`python_logic_routing=true`, with `NERAL_EXPERIMENTAL_TIR=1`
configured on the local server.

The route bypasses the narrow natural-language-to-Z3 schema extractor for that
specific request. Qwen3.5:9B chooses a single `NERAL_EXECUTE_JSON` proposal,
with a new nonthinking **768-token** first-decision limit and existing
`num_ctx=4096`. The generated program is executed **only** in the existing
offline, read-only Docker sandbox, which has its own CPU/memory/time limits.
A candidate must emit a JSON object `{"solutions":[...]}`; any count other
than 1, a malformed packet, failed execution or malformed output falls back
to unchanged raw Qwen. No benchmark-string matching or new RegEx trigger is
used. Default TIR and ordinary chat do not change.

Even for a singleton result, the generated program may **omit or misread a
natural-language clue**. Execution and uniqueness are not semantic proof.
The new code returns `tool_observed_unverified`; the PR #537 conservative
guard MUST choose the original raw Qwen response instead of promoting it.
This pilot only tests reliable tool selection and candidate discovery, **not**
a verified intelligence gain or a promise of zero generalization regressions.

## Exact known integration fixture, not a blind test

The 1–5 code task has one independently enumerated valid code: `21453`.
The CPU fixture proves the expected combinatorial result, checks model-proposed
tool syntax with a mocked model and sandbox response, and tests malformed,
empty and multiple-solution fallbacks. **It does not prove GPU inference
success on a live NERAL installation.** Treat this fixture as known/contaminated
for all future release scorekeeping.

## Latency hypothesis

`num_ctx: 4096` and `keep_alive: 5m` were already set for Ollama. Simply
setting the same context length again cannot explain or fix 118 model swaps.
Measure `load_duration`, `eval_count`, `prompt_eval_count`, wall-clock
latency, model IDs and the change in resident VRAM over paired runs; never
upload private prompts or generated code. A shorter tool-decision decode may
reduce tokens, but the extra review and guarded raw fallback can increase
total latency. No latency gain is claimed without a controlled GPU A/B.

## GPU & release gate

Existing self-hosted GPU workflows authorize inference only on exact,
release-attested `main` SHAs; a PR branch must **not** bypass those checks.
After review and explicit GPU authorization, use one separately scoped,
non-release pilot on the actual RTX 4070 to measure tool-call validity,
the number of token-limit failures, completion/accuracy for the known
`21453` case, exact model digest and elapsed time.

Only then create an independently sourced, untouched holdout with 100 tasks,
saved answer keys and fixed paired seeds before the run. Compare raw
Qwen vs experimental guarded route at identical settings. Treat
`>=8` paired wins and 0 observed losses as an **observed gate**, never
a statistical guarantee of zero regression. Do not promote model-authored
logic scripts as verified solely because `len(solutions)==1`.

No model training, cloud APIs, hidden data uploads or changes to `main`
are authorized by this draft.
