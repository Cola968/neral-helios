# NERAL THINK: verified Python-program tracing (safe local tool)

**Purpose:** extend the already installed local THINK mode from exact arithmetic
to a bounded subset of Python output questions. This is an algorithmic
improvement of the *assistant system*, **not** a claim that Qwen / v5 / v6
neural weights gained reasoning ability.

## What is implemented

- The new `neral_code_trace.py` parses input with `ast.parse`, never
  `eval`, `exec`, `compile`, subprocess, shell, `pickle`, network,
  file I/O, arbitrary Python function calls or reflection.
- Explicit allowlist for small integer arithmetic (+,-,*,//,%), 1–3 argument
  bounded `range`, finite `for` loops, branching and `continue`,
  named assignments and integer `+=` / `-=`, lists, integer indexing,
  slicing, short strings, dictionaries, `sum`, `len`, `count`,
  `dict.values`, and **one final integer `print`**.
- Fails closed on unsupported syntax, excessive operations, large numbers,
  oversized sequences, large source input, unknown names, dangerous
  builtins and ambiguous output.
- `neral_think.think_reply` only recognizes a fixed set of *whole,
  single-turn* German/English output questions with exactly one fenced
  `python` block. Multi-turn requests, unsanctioned prompts, broad code
  reviews and arbitrary user code do not silently invoke this tool.
- If NERAL's first model answer is unambiguous and exactly correct according
  to the restricted interpreter, response is marked `model_generated=true`.
  If incorrect or ambiguous it gets **at most one** retry; after failure the
  tool's independently computed answer is returned with
  `model_generated=false`, `verification_status=tool_fallback` and
  `tool_used=neral_code_trace_safe`.
- In particular a tool-derived correct value **must never** be presented as
  improved LoRA generalization.

## Independent CPU validation

`python -m unittest discover -s tests -p test_neral_code_trace.py -v`

Tests deliberately use **new examples, not NERAL's consumed dev72 prompts**:
six new representative Python programs; 40 generated programs across
two new parameterized sets; 20 negative security cases; prompt-injection /
multiple-code-block refusals; and THINK-integrated correct/wrong-model
fakes. This is proof of implementation behaviour on those cases, not a
GPU benchmark, not functional coding generation, not random held-out model
inference and not a release-grade intelligence metric.

`python -m unittest discover -s tests -p test_neral_think.py -v`

The existing maths behaviour and single-turn/verification constraints must
remain unchanged. Both test groups are required in the CI.

## Follow-up for model-weight improvement

1. Examine the **private** original 19 non-parseable v6 dev72 answers using
   the existing CPU-only `neral-v6-dev72-regression-cpu-manual.yml`
   diagnostics. Never upload raw private outputs.
2. Generate independently verified **train-only** Python semantics and
   mathematics examples from diversified ASTs/oracles. Exclude dev72 and
   release cases, including near duplicates. Mix with v5 replay to guard
   against catastrophic forgetting; manually inspect actual sources.
3. Evaluate fresh **unseen** code-writing, code-tracing, math and reasoning
   tasks against Qwen+Few-Shot, v5, and any new candidate using identical
   prompts, decoding and budgets. Tools-on versus tools-off must be scored
   separately. Promotion is subject to `GATE.md`; do not claim improvement
   from optimizer steps or token loss.

References: <https://karpathy.ai/zero-to-hero.html> and
<https://huggingface.co/docs/trl/sft_trainer>. The latter documents that
completion-only loss and validation support training hygiene, but neither
by itself proves accuracy on unseen tasks.
