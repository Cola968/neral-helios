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

Suppose the question says: **X <= 2 and X <= 1**, but *both* extractors encode only **X <= 2**. The original number 1 still appears in the declared domain, and the <= operator is still present, so this also passes the lexical coverage gate. Z3 correctly proves both **wrong formulas** equivalent; the candidate count is 3 rather than the correct 2. This is an explicit regression test and is never called verified. A shared base model, shared training data, similar prompts, or common quantifier mistakes may produce correlated errors. Keyword-and-number checks are similarly not semantic proofs. To promote NL answers, introduce an independently trusted **question-to-formula completeness certificate** or evaluate a preregistered, independent blind corpus with an explicitly accepted risk policy. An observed raw-Qwen `done_reason=length` does **not** prove a proposed solver answer correct or guarantee zero regressions.

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


## Local RTX 4070 live diagnostics extension (still not executed)

The default `scripts.neral_nl_schema_probe` remains concise and opt-in. New `--diagnostics-file` explicitly writes **raw model JSON responses** to a **local file**, one case per run. It measures elapsed API round-trip time and records Ollama's `load_duration`, `total_duration`, `prompt_eval_duration`, `eval_duration` (nanoseconds converted to milliseconds), `prompt_eval_count`, `eval_count`, and `done_reason` where provided. Parsing, exact model counting, source coverage, Z3 consensus and each model's old-case reference count are reported separately. A matching reference count does not certify semantic fidelity. The script runs two independent Ollama requests even when the first candidate fails. No upload to GitHub is performed.

In **Windows PowerShell on the machine that actually runs Ollama**:

```powershell
git fetch origin
git switch feat/neral-nl-schema-consensus-bridge-20261112
ollama list
ollama ps
nvidia-smi
python -m pip install "z3-solver>=4.13,<5"

# Replace with the TWO EXACT installed tags from 'ollama list'.
$QLORA = "YOUR_LOCAL_QLORA_OLLAMA_TAG"
$BASE = "YOUR_LOCAL_BASE_QWEN35_9B_TAG"

foreach ($case in @("21453", "logic-02", "logic-06")) {
    python -m scripts.neral_nl_schema_probe --case $case --live --qlora-model $QLORA --base-model $BASE --diagnostics-file "nl-schema-$case.json"
}
```

A `nonzero` process exit is **expected** when the candidates are invalid, non-equivalent or otherwise inconclusive. Check the locally created JSON file even if the CLI returns a failure status. Choose a fresh filename for repeats: the CLI refuses to overwrite existing evidence. A QLoRA checkpoint not imported into Ollama does not qualify as an installed Ollama tag; failing to find the model is a setup problem, not evidence against QLoRA's ability.

Interpret timing carefully: Ollama's `load_duration_ms` counts loading, and `elapsed_ms` is the local API round trip for that individual extraction, **not** the entire multi-call inference pipeline. Confirm actual GPU offload with `ollama ps` (PROCESSOR) and `nvidia-smi`; the presence of a GPU by itself does not prove that Ollama used it. Environment effects like warmup/cold-load matter before latency claims.

**Disclosure:** The diagnostic JSON includes unredacted generated responses from the three intentionally public historical tasks. Review locally before sharing. Never run this diagnostic mode on private prompts or on the sealed next-100 corpus. No second-100 files, hashes, or samples are required by this command.

**Promotion policy:** Passing three historical tasks is useful for feasibility, but insufficient to open the prospective-100 release gate. Training QLoRA v2 is not justified solely by one failed prompt or malformed JSON. First compare the base model, test one schema prompt/repair variant on *development-only* old tasks, and evaluate whether a repeatable adapter-specific gap remains.


## Structured Outputs repair after first RTX 4070 pilot (2026-11-12)

The user reported that all six model invocations across 21453, logic-02 and logic-06 failed prior to Z3 consensus. The old 768-token, generic \`format:"json"\` requests often produced malformed typed v2 ASTs, wrong version strings, list-valued variables, free-text logic and repetition/length truncation. These observations motivated this **isolated** development-only change; they do not prove that a new adapter is needed.

### Changes

- \`neral_nl_schema_format.py\` defines a **real JSON Schema object** matching typed v2's grammar: integer \`version:2\`, four exact top-level fields, identifier-keyed \`variables\` objects with strictly typed integer/Boolean domains, \`constraints\` as recursively typed predicate ASTs, and \`query\` restricted to \`count\` or \`satisfiable\`. \`oneOf\` and \`$defs/$ref\` encode \`compare\`, \`all_different\`, \`exactly\`, Boolean connectives, \`var\`, \`const\`, \`sum\` and \`mod\`.
- All three old-case local Ollama model invocations now send \`format: get_v2_ollama_format()\` (a schema object rather than the string \`json\`), \`num_predict:1024\`, \`repeat_penalty:1.05\`, \`temperature:0\`, \`think:false\`. Both extraction paths use the same grammar specification but retain **different** instructions and distinct installed Ollama model tags.
- Prompts explicitly use integer palette values for graph colors, specify \`variables\` as a map instead of a list, prohibit free-text \`logic\`, describe all AST operators, and discourage duplicate redundant clauses. Sampling parameters are experimental hypotheses, **not** measured quality improvements.
- The existing advisory \`coverage_gate\` now also checks narrow, recognizable source conditions with individual AST witnesses: each explicitly listed graph edge, red vertex zero, exact Boolean cardinality, opposing first two bits, every forbidden distance-two pair, and the numbered German PIN's all-different, nonadjacent order, single-digit-between, immediate-right and non-edge clauses. Unrecognized wording receives only lexical checks; even passing every witness is **not** a semantic truth certificate. Alternative equivalent representations may be rejected. Result remains \`model_generated:true\`, \`verified:false\`, \`safe_to_promote:false\`.
- The original typed solver, \`neral_tir_regression_guard.py\` (PR #537), main, adapter weights, training pipeline and sealed prospective-100 dataset SHA \`3feca67d6ee9fd48b462bdbf3e15e256b9e72a1a35357d30d75d383b9a7b280d\` remain unchanged.

### What Structured Outputs does *not* establish

A JSON Schema can constrain generation and identify a syntactically valid object; it **cannot** prove all semantic constraints were translated, prevent all possible model/runtime bugs, or guarantee that duplicated raw JSON members can never appear under all sampler versions. The strict downstream v2 parser still rejects duplicate keys and invalid types. Ollama's support for the recursive \`$defs/$ref\`, \`oneOf\` and \`patternProperties\` combination must be verified empirically against the **installed local Ollama server**. An unsupported grammar/runtime error must fail closed; do not silently switch back to unconstrained JSON or promote a candidate.

### CPU verification / re-run

[CPU run #43302848266](https://github.com/Cola968/neral-helios/actions/runs/43302848266) passes 73/73 focused and regression tests, including the existing three historical case fixtures, novel deliberately omitted source clauses, JSON Schema draft-2020-12 validation, and fake Ollama request assertions. **All tests were CPU-side or fake model responses, not real local inference.**

For the second three-case RTX 4070 check, on Windows PowerShell in the exact PR #542 branch:

\`\`\`powershell
git fetch origin
git switch feat/neral-nl-schema-consensus-bridge-20261112
python -m pip install 'z3-solver>=4.13,<5'
ollama list
ollama ps
nvidia-smi
$QLORA='neral-qwen35-qlora-v1:latest'
$BASE='qwen3.5:9b'
foreach ($case in @('21453', 'logic-02', 'logic-06')) {
  python -m scripts.neral_nl_schema_probe --case $case --live --qlora-model $QLORA --base-model $BASE --diagnostics-file "structured-v2-$case.json"
}
\`\`\`

Validate those exact model tags first with \`ollama list\`, and check the runtime offload with \`ollama ps\` and \`nvidia-smi\`. Each report is a new, unredacted **LOCAL** JSON file; the script will refuse to overwrite a previous report. Nonzero exit statuses are expected for invalid candidates or non-consensus. Inspect \`raw_json\`, \`status\`, \`coverage_reason\`, \`done_reason\`, \`load_duration_ms\`, \`total_duration_ms\` and \`elapsed_ms\` in each local file; do not publish raw files without reviewing them.

Passing the three already-known tasks only supports moving to a **separate, preregistered evaluation decision**, not automatic use of the sealed prospective set or a proof of zero regressions. Compare actual syntax acceptance, source condition coverage, correct reference counts, raw lengths and load times between base and QLoRA first. Decide on a new training run only after a reproducible adapter-specific gap remains despite constrained decoding and bounded prompt changes.


## Source-bound nonempty variables and three complete few-shots (2026-11-12)

Second live RTX 4070 results **as reported by the user**: Ollama accepted the previous recursive schema and all six responses finished \`done_reason=stop\`, but every response had \`"variables":{}\`. This is an observed \`minProperties\` grammar enforcement gap, **not** evidence that the language models cannot reason in constraint logic.

The opt-in local probe now uses \`get_v2_ollama_format(question)\`. For three **generic**, independently recognizable source formats (not benchmark IDs), it builds named variable declarations before inference:

- \`Color vertices 0..N with red, green or blue\` → \`C0\` … \`CN\`, integer domain \`[0,1,2]\`.
- \`binary strings of length N\` → \`B0\` … \`B(N-1)\`, Boolean domain \`[false,true]\`.
- \`Ein N-stelliger Code verwendet 1,2,...N genau einmal\` → \`P1\` … \`PN\`, integer positional domain \`[0,...,N-1]\`.

These declarations are **deterministic scaffolding derived from the original question**, not model-authored constraints, verified answers, gold labels, or hidden dataset lookup. N is bounded at 10. Each recognized schema uses explicit \`properties\` and \`required\` keys with \`additionalProperties:false\` for its variables object. It no longer needs the ignored \`minProperties\` to reject an empty map at generation time *if that local Ollama sampler honors required object keys*. For unsupported or ambiguous wording, the original generic schema stays in place; the strict typed solver still rejects empty maps. **This does not guarantee that all arbitrary future prompts yield nonempty declarations.**

Both independent system prompts now contain three complete **synthetic**, solver-validated JSON examples different from the three historic measured tasks: a small three-color graph, a four-bit count with every forbidden distance-two pair explicitly unrolled, and a four-digit positional ordering problem. The accompanying rules explain that adjacency indices run \`i=0..n-2\` while forbidden distance-two pairs run \`i=0..n-3\` (inclusive), to avoid off-by-one omissions. The two model prompts remain distinct, and each model receives only the same original question plus its own system prompt; they never see the other's output.

The local diagnostic status \`empty_variables\` now separates this precise error from generic \`invalid_v2_schema\`. No automatic self-repair, hidden rerouting, re-training, silent fallback to \`format:"json"\`, or verified model-generated answer has been introduced. \`neral_tir_regression_guard.py\` (PR #537) remains unchanged.

[CPU verification #43307104994](https://github.com/Cola968/neral-helios/actions/runs/43307104994): **81/81 tests green** across existing NL bridge, old historical cases, diagnostics, source-bound variable generation, complete few-shots, typed solver, and guard. This does **not** constitute new model inference or a hardware-vetted constrained-sampling result.

### Next local GPU check (Codex or PowerShell)

Use the exact existing two Ollama tags, after pulling the **latest PR #542 head**. New filenames avoid overwriting the last experiment:

\`\`\`powershell
git fetch origin
git switch feat/neral-nl-schema-consensus-bridge-20261112
git pull --ff-only origin feat/neral-nl-schema-consensus-bridge-20261112
ollama list
$QLORA='neral-qwen35-qlora-v1:latest'
$BASE='qwen3.5:9b'
foreach ($case in @('21453','logic-02','logic-06')) {
  python -m scripts.neral_nl_schema_probe --case $case --live --qlora-model $QLORA --base-model $BASE --diagnostics-file "fewshot-v2-$case.json"
}
\`\`\`

Inspect both raw schemas, whether \`required\` produced nonempty declarations in practice, missing constraints, reference counts, model total/load durations, and consensus without promoting any unverified candidates. Keep the prospective 100-case sealed corpus completely untouched. No merge to main.
