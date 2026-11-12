# NERAL: dual NL-to-v2-schema consensus (development-only, 2026-11-12)

## Scope, provenance, and hard boundary

- Stacked on `feat/neral-typed-constraints-v2-20261112`, not merged to `main`.
- Uses the existing, exactly bounded v2 finite-domain solver without changing its source, hard caps, QLoRA weights, or default router.
- Does **not** touch or score the sealed next-100 dataset SHA256 `3feca67d6ee9fd48b462bdbf3e15e256b9e72a1a35357d30d75d383b9a7b280d`.
- Never labels a **model-translated** schema `verified=True`. PR #537 retains authority: an unverified TIR response cannot replace the unchanged baseline Qwen response.
- No GPU/model inference in GitHub Actions. The CPU fixtures simulate two independently invoked extractors. They do not test QLoRA or base-Qwen extraction accuracy.

## Algorithm (CPU only by default)

1. Two separate caller-provided extractors receive the same original question but **different system instructions**. They never receive each other's candidate. The caller must supply two genuinely separate model backends if it claims a model-comparison experiment; Python callback identity alone cannot prove that.
2. Each response must be a single complete typed v2 JSON object. The existing strict solver validates all types, domains, constraint ASTs, hard limits and returns a temporary count that is **not** released to the user.
3. The variable names, finite typed domains and query type must agree exactly. A renamed variable or different range cannot silently pass as equivalent.
4. An advisory **coverage gate** looks for overt missing digits, capital-letter variables, relation symbols, and selected broad English/German logic markers. It is conservative and may reject valid paraphrases. It cannot establish that all conditions were captured: identical omissions and reversed interpretations can pass lexical checks.
5. Z3 adds identical finite-domain restrictions and asks if the XOR of the conjunctions is satisfiable. Only `UNSAT` establishes equivalence **between the two proposed formulas**. `SAT` is a real counterexample to formula equivalence; `unknown` / timeout / internal Z3 errors are inconclusive. Without Z3, exact enumeration may be used only if the full domain product has at most 100,000 assignments.
6. The bridge returns a *candidate* number with `status=consensus_unverified`, `schema_equivalent=true`, `verified=false`, `model_generated=true`, and `safe_to_promote=false`. It does not emit the guard-trusted formal-user-schema signature.

### Why two matching models do not prove correctness

Suppose the question says: **X != 1**, but *both* extractors omit that constraint. Z3 correctly proves both **wrong formulas** are equivalent; that says nothing about the original question. A shared base model, shared training data, similar prompts, or common quantifier mistakes may produce correlated errors. Keyword-and-number checks are similarly not semantic proofs. To promote NL answers, introduce an independently trusted **question-to-formula completeness certificate** or evaluate a preregistered, independent blind corpus with an explicitly accepted risk policy. An observed raw-Qwen `done_reason=length` does **not** prove a proposed solver answer correct or guarantee zero regressions.

## Established old-case CPU checks

- `logic-02`: known graph coloring case from `scripts/neral_blind_100_cases.py`, manually constructed schema; equal count to existing ground truth.
- `logic-06`: old binary-string distance-two constraints; same process.
- `21453`: exact German PIN wording from the already-existing PR #539 reference. A separately implemented enumeration confirms that the only actual string is `21453`. The v2 *count* is 1, not a generated PIN itself.
- Negative tests: distinct solution sets, redundant but equivalent constraints, mismatching domains/query types, omitted numeric anchors, duplicate JSON keys, incomplete responses, model exceptions, Z3-unknown and strict PR #537 fallback. No generated schema may bypass the guard.
- `.github/workflows/neral-nl-schema-bridge-cpu.yml` runs the new tests plus v2/guard/Boolean regressions on CPU with Z3. A green run is structural proof **of code behavior**, not model accuracy.

## Optional manual *real model* probe (not executed in CI)

Only when the user has installed two *different* local Ollama tags, including a QLoRA model exported/imported separately (a Hugging Face adapter file alone is **not** an Ollama model tag):

```bash
python -m pip install 'z3-solver>=4.13,<5'
python -m scripts.neral_nl_schema_probe --case logic-02
python -m scripts.neral_nl_schema_probe --case logic-02 --live --qlora-model YOUR_LOCAL_QLORA_TAG --base-model YOUR_LOCAL_QWEN_TAG
python -m scripts.neral_nl_schema_probe --case logic-06 --live --qlora-model YOUR_LOCAL_QLORA_TAG --base-model YOUR_LOCAL_QWEN_TAG
python -m scripts.neral_nl_schema_probe --case 21453 --live --qlora-model YOUR_LOCAL_QLORA_TAG --base-model YOUR_LOCAL_QWEN_TAG
```

- Without `--live`, the CLI only reports `live_not_requested`. It performs **zero inference**, even locally.
- The live CLI calls only `http://127.0.0.1:11434/api/chat`; there is no remote endpoint or implicit model download. User questions and untrusted proposals stay local; output is limited to short status labels, a candidate count, and proof backend.
- `think=false`, temperature 0 and 768 output tokens are explicitly set for both, but this does not make their quality or latency equivalent.
- A result of `consensus_unverified` is **still unverified**. Model loading and memory footprint must be measured on the RTX 4070 and compared separately.

## Deliberately outstanding

- Real QLoRA-versus-base extraction acceptance and answer quality, including false agreement rate, have **not** been measured.
- The requested `fast-old6` GPU telemetry is not in run `#43282592123`, which only sealed the prospective dataset. Proof-First remains **not** the default. Need an actual successful GPU run/artifact before latency, phase and model-load comparisons.
- No switch from the PR #537 raw-Qwen fallback on naturally worded queries; no merge to `main`; no use of the sealed prospective set for tuning.
