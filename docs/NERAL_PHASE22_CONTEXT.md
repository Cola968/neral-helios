# Phase 2.2: complete history and priority-preserving context

## Operator-authorized rework, 2026-11-02 (Windows CPU)

Attempt 2 follows the operator's explicit rework approval for #223 and stopped
executor/stable-head checks. Eleven new cases first failed for the missing
completion/status/provenance/estimate contracts. Three additional source-schema
cases failed before requiring safe relative project paths and SHA-256 syntax.

Assistant history now requires `completion_status="complete"` and
`provenance_verified=True`; missing legacy metadata is rejected. The latter is
trusted-caller evidence, not proof supplied by a browser boolean. Helios must
verify a server-issued receipt and set it itself; an untrusted request must not
be passed through with this field. Runtime/family identity checks still apply.
No completion flag certifies factual correctness. Test fixtures for retained
history now declare completion/provenance; rejection tests start from otherwise
valid rows, so they still exercise each separate invalid marker.

Controller results expose the five-valued `completion_status`, measured provider
`cancel_ack_latency_ms` (or null), and a `context_budget` object. Failure and
unavailable statuses cannot look like complete history. Questionable unsupported
test-execution claims remain incomplete and require review. Invalid/nonfinite
latency metadata fails closed. Transport/server closure provenance stays separate.

For Ollama Qwen3.5, absent preflight token counts use UTF-8 serialized message/tool
bytes plus 256 safety units and 32 per message. This intentionally pessimistic
heuristic includes system and schema content; `conservative_estimate=True` always
labels it, and it never changes observed input/output usage. It is not a measured
server tokenizer count or a certification of arbitrary custom Ollama templates.
The fixed normal Ollama context limit is 4096; local Qwen retains its exact
counter and 2048 limit. Mandatory prompt overflow fails before generation.
Tool results still outrank project records, which now serialize an explicit
`untrusted_project_content=True` label and require relative path/digest metadata.
Digest format alone is not source authenticity; trusted readers supply it.

Current Windows module suite: **296 passed, 2 skipped, 2 deselected**. Deselections
are the unchanged fixtures requiring symlink privilege and POSIX O_NOFOLLOW;
no privilege or OS setting was changed. A temporary source composition with
transport #225 at `85c4f73` passed **244 tests, 1 deselected**, confirming its
provider/transport works with this context/controller. These are overlapping
scopes, not additive totals. The temporary check did not merge branches or alter
another task's files. No model inference/training/download/weight changes,
paid services, merge or deployment occurred.

The following behavior/test notes describe the earlier Linux commit; the new
strict completion policy supersedes its legacy-metadata allowance.

Base: PR #222 at `c4c95d5ff2749cd544374f878621e84a2601b937`.
Task: `neral-phase22-context-v1`. Source changes are confined to the leased
context/controller files and their existing tests. No legacy runtime, stores,
UI, model weights, training data or product loading path changes.

## Behavior

- The controller rejects history with explicit non-completion evidence before
  calling a provider. The context selector omits the entire corresponding
  user/assistant pair. Evidence includes incomplete, interrupted, cancellation,
  timeout, `complete=False`, unsuccessful status and non-terminal finish reasons.
  Boolean metadata is validated strictly; `complete=0` is not accepted as a
  completion flag. Legacy rows without completion metadata remain supported;
  their acceptance is not an independent verification of factual correctness.
- Context records retain current caller-supplied tool results before stored
  decisions and raw project files. Complete old history pairs are omitted first.
  Supplied tool records are a trusted-caller API input, not a mechanism for
  promoting model-proposed text to verified evidence. Their content remains
  explicitly untrusted reference data.
- Exact counters now select history even when no project record or decision is
  supplied. A provider without a reliable counter can still use the bounded
  existing basic history path, but reference enrichment fails closed.
- Before a follow-up local Qwen call, the current assistant/tool message protocol
  is included in the exact count and retained while older history/references are
  omitted. Tool envelopes are never orphaned. If current task plus system/tool
  overhead and current tool protocol cannot fit, the call fails closed. The
  current request is not silently shortened and the local limit stays 2048.
- Controller results distinguish cancellation requested/acknowledged, transport
  closed and observed server execution stopped. Unknown transport/server facts
  stay null. Provider timeout reports `timeout`, preserves actual reported usage
  and stops before any proposed tool execution. Provider fields are consumed
  compatibly so this controller change also works before the transport PR lands.

No persistent memory, new relevance model, project-version authenticity or native
Windows safety proof is introduced by this correction.

## Test evidence (Linux CPU)

Tests were written and run before fixes:

- Initial defects: 28 failed, 57 passed (`/tmp/phase22-context-red.txt`).
- Current tool protocol under second-call budget pressure: 1 failed
  (`/tmp/phase22-context-tool-red.txt`).
- Strict completion metadata: 6 failed, 92 passed
  (`/tmp/phase22-context-strict-red.txt`).

Final existing module suite: **280 passed, 1 native Windows skip**, 7.13 seconds
(`/tmp/phase22-context-modules.txt`). This is the original 239 plus 41 new tests;
parameterized cases are counted as separate executions. The suites overlap and
are not additive proof of distinct product capabilities.

Final unfiltered repository suite: **890 passed, 2 skipped, 5 failed**, 15.56
seconds (`/tmp/phase22-context-full.txt`). All five failures are the same missing
ignored data/checkpoint artifacts already present at the exact unchanged base:

- `tests/test_curriculum_v2.py::test_v2_splits_and_hashes`
- `tests/test_curriculum_v2.py::test_pilot02_lineage`
- `tests/test_growth.py::test_40m_exact_count_and_inherited_weights_unchanged`
- `tests/test_regressions.py::test_curriculum_disjoint_and_masked`
- `tests/test_regressions.py::test_original_untouched_and_parameter_count`

An unchanged `benchmark_gsm8k50.py:32` also emits an invalid escape-sequence
SyntaxWarning. No test-only artifact was generated or downloaded to conceal a
failure. Local HTTP/browser fixtures ran with local socket permission, not a
model provider. Native Windows/NTFS and GPU tests remain unavailable acceptance
gates. No language-model inference, training, paid job, merge or deployment ran.

Commands use `PYTHONDONTWRITEBYTECODE=1 HELIOS_DEVICE=cpu CUDA_VISIBLE_DEVICES=''`
and `/tmp/neral-control-check/bin/python -m pytest -p no:cacheprovider -q`, with
`tests/neral_control` for the module suite and no test filter for the full suite.

## Attribution and release status

These are correctness improvements in NERAL's context and control layer. They
are not learned model-weight improvements and do not establish a 20% task gain.
The change is CPU-tested, not GPU-tested, not quality-benchmarked and not a
release approval. Actual adapter results and server-side Ollama GPU-stop evidence
remain separate requirements.
