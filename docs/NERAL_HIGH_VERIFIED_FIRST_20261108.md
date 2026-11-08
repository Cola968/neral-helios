# NERAL HIGH: Verification-first (08.11.2026)

## Problem

The local HIGH route previously generated with Qwen3 before using its existing
verified arithmetic/logic/Python trace tools. For fully recognized numeric
questions this wasted GPU tokens and could generate an incorrect intermediate
answer. This is different from the open-ended reasoning problem.

## Implemented on PR #418

- Only a trusted local `OllamaLocalBackend` serving HIGH uses the preflight.
- Fully recognized standalone arithmetic/logic/Python-output inputs can return
  an exact tool answer with `calls_used=0`, `verified=true`,
  `model_generated=false`, `verification_status=tool_direct`, and a tool ID.
- A request for worked steps/explanation, an incomplete or ambiguous input, and
  any multi-turn context are sent through the old HIGH path unchanged.
- The safe math AST, finite-logic solver and non-executing Python AST tracer
  are reused. No `eval`, `exec`, shell, external provider, auto-training,
  model weight mutation or automatic adapter promotion.

## Evidence

The Linux CI suite for `tests/test_neral_high_verified_first_probe.py`
recomputed independent integer references for 24 seeded, new *parameters*
across modular arithmetic, directed graph reachability and static Python
tracing. The narrow exact preflight matched all 24 independently computed
values. This is a correct tool-assisted system feature, **not** evidence of
better learned neural weights or generalized understanding on unfamiliar
method families.

The paired REAL HTTP LOW-vs-HIGH job is separate and scheduled only when
the Windows self-hosted RTX-4070 runner is free. Its scalar-only report
contains per-domain scores and latencies, model digest, paired wins/losses,
and explicit labels saying HIGH got tools and LOW did not. It does not
upload prompts or model text. CPU success is NOT a substitute for a GPU
comparison or a release gate.

## Research and release constraints

1. Publish separately: NERAL neural model (Qwen base vs NERAL LoRA with
   equivalent tools/decoding), product system (LOW vs HIGH), latency,
   verifier coverage, and error categories.
2. Do not claim NERAL surpasses Qwen generally from narrow fixed-parser
   tasks; Qwen equipped with the same verifiers might be equally accurate.
3. Qualifying future results require a frozen, unseen multi-method suite,
   independent grading, and no relevant regressions.
4. PR #418 targets the in-progress LOW/HIGH PR #403, not stable main.
