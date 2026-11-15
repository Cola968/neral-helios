# NERAL 1.1 Adaptive Planner – Opt-in THINK integration

Implemented only on feature branch feat/neral-1.1-adaptive-planner.
Do not merge, release, train or launch an RTX4070 GPU benchmark without
separate review and authorization. Original PR542 commit 737413f0 and
main 7fb05452 remain unchanged.

## Enabling the experimental HTTP route

Requires both:
1. Local environment variable NERAL_EXPERIMENTAL_ADAPTIVE_PLANNER=1.
2. Explicit request JSON key "adaptive_planner": true for
   POST /api/neral/chat/think.

Example JSON body:
    {"prompt":"Count permutations of numbers 1 through 4. Number 1 is before number 2.","adaptive_planner":true}

This is an example of the supported strict source grammar, not a blind-test case.
The normal THINK endpoint, ordinary chat, and prior experimental flags are not
altered unless this exact new flag and environment setting are both present.

The route is rejected for incompatible flags, non-boolean values, or anything
except a local OllamaLocalBackend using qwen3.5:9b and
measured_nonthinking_2048. The existing synchronous chat busy lock still applies.

## Model and sandbox execution

- Candidate model: pinned local model TAG neral-qwen35-qlora-v1:latest.
- Each request of that model includes the strategy and tool payload together,
  not a separate planning-only model pass. A bounded structured JSON response
  must end with done_reason=stop. Incomplete replies are not executed.
- Strategies: Z3_SCHEMA / PYTHON_SANDBOX / DIRECT.
- The deterministic prefilter uses no benchmark case IDs and directs simple
  conversational requests to untouched Qwen immediately.
- The proposer uses only the existing localhost-only nonredirecting Ollama
  transport, with no proxies or remote-model fallback.
- Python proposals run via the existing network-disabled, read-only Docker TIR
  sandbox, never host Python. Only positive independent full-source proofs can
  be promoted; successful execution by itself is insufficient.
- For the currently supported narrow, structured English combinatorics tasks,
  the trusted Python verifier enumerates every bounded assignment independently
  and compares the output exactly. Unknown source clauses mean no promotion.
- For a strictly parsed v2 source certificate, the existing PR542 coverage
  and exact Z3 equivalence guard remains authoritative.
- At most two repair turns after the first attempt. Feedback contains only
  enumerated error codes; never traces, model answers, dataset golds or raw text.
- Unsupported, unavailable or unverified results invoke the original PR537
  guarded raw Qwen fallback on an untouched copy of the original messages.
  Unusable Qwen fallback gives an explicit unverified refusal.
- Multiturn history is not eligible for full-source verification; it stays
  on the Qwen fallback without calling the adaptive proposer.

## Verification and limitations

The 15 November CPU-only CI run 43740946562 passed 194/194 tests: 19
controller, 19 Python-verifier, 7 local-proposer transport, 9 Flask-HTTP,
3 offline-aggregate and 137 prior PR537/542 regression tests.
The existing THINK CI workflow also succeeded at 43740946544.

No GPU inference was performed in this step. The implementation does NOT
demonstrate higher accuracy, lower latency, improved general intelligence,
or a universal zero-regression guarantee. It does not prove the model will
actually emit well-formed useful plans on the RTX4070.

The next acceptance stage should check actual local GPU wiring and first-turn
candidate quality on isolated non-blind synthetic smoke tasks. A **new**
independently seeded, sealed 100-task benchmark should follow only after
a CPU and GPU smoke gate is complete, using a strict untouched-Qwen
comparison. Never re-use the previous Blind100 data to optimize routing.

The 85/100 objective remains unproven.
