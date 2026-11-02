# Phase 2.2: complete history and priority-preserving context

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
