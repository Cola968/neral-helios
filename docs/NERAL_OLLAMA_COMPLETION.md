# Qwen completion integrity — 1 November 2026

## Reproduced defect and smallest fix

On PR #210 head `9ba549d3e014efd685fb645b84bec7ded56a2162`, a simulated
Ollama response with `done=true`, `done_reason="length"` and nonempty content
was returned as an ordinary complete answer. The browser consequently saved it
without `interrupted=true`, and could reuse the unfinished pair as model history.
The development gate could even pass an explicitly truncated correct prefix.

`neral_assistant.py` now preserves the finish reason and returns
`incomplete=true` for `length`. The existing browser rendering, persistence and
context exclusion handle that flag; no second history implementation was added.
A truncated tool request is refused before either allowlisted tool runs.
Unknown or malformed explicit reasons fail closed. No retry or automatic token
budget increase is introduced. A textless response still fails the existing guard.

Ollama's [chat API](https://docs.ollama.com/api/chat) separates `done` and
`done_reason`. Its [response type](https://github.com/ollama/ollama/blob/main/api/types.go)
allows omission of `done_reason`. For compatibility, an absent field retains
the prior `done=true` behavior and is reported as `finish_reason=null`;
absence is not independent evidence of EOS. Explicit null/empty/unknown values
are rejected. This change cannot detect undisclosed upstream truncation.

The existing 2022 development gate records a known truncated response as
`incomplete`, counts automatic cases as failures and keeps manual cases pending.
It continues to later cases. Correct-looking text cannot override truncation.

## Verification

All model responses below are simulated; no actual Qwen or Arctus inference,
training, downloads, checkpoint changes or GPU work was performed.

- RED before production changes: 8 failures, 32 passes across provider, Flask
  and gate tests. Failures included actual tool execution before rejection,
  missing finish metadata and a false positive development-gate pass.
- GREEN for the same focused command: 40 passes.
- Full CPU contract selection used by `arctus350-flask-cpu.yml`, plus gate:
  **156 passes**. Four socket tests initially failed under the sandbox's
  loopback restriction; the permitted local HTTP run passed all 156.
- Node history/provenance selection: **24 passes**.
- Real headless browser -> Flask -> fake Ollama: **PASS**, including the
  truncated-answer label, reload/persistence and exclusion of that pair from
  subsequent provider messages. Normal stop, thinking, provenance and offline
  handling remain covered. The fake server now uses an ephemeral port, patched
  only inside the test subprocess; it never contacts the user's real Ollama.
- Bare `pytest -q --tb=no`, CPU and empty assets: **526 passed, 1 skipped,
  5 failed**. The remaining failures require absent legacy private artifacts:
  `test_curriculum_v2::test_v2_splits_and_hashes` (curriculum manifest),
  `test_curriculum_v2::test_pilot02_lineage` (pilot_02 checkpoint),
  `test_growth::test_40m_exact_count_and_inherited_weights_unchanged`
  (neral40m checkpoint), `test_regressions::test_curriculum_disjoint_and_masked`
  (curriculum train data), and
  `test_regressions::test_original_untouched_and_parameter_count` (seed checkpoint).
  None were disabled, generated or downloaded. The restricted bare run also
  had four socket failures and an existing `benchmark_gsm8k50.py` escape warning.

Local commands use the existing Python environment, `HELIOS_DEVICE=cpu`,
`CUDA_VISIBLE_DEVICES=''` and empty `HELIOS_ASSETS_ROOT`:

```text
python -m pytest -q tests/test_neral_assistant.py tests/test_neral_2022_gate.py tests/test_issue195_assistant.py
python -m pytest -q tests/test_arctus_flask.py tests/test_helios_runtime.py tests/test_helios_chat_experience.py tests/test_helios_workspace.py tests/test_helios_i1_ready.py tests/test_helios_install.py tests/test_issue195_assistant.py tests/test_neral_assistant.py tests/test_neral_assistant_code_tool.py tests/test_neral_assistant_launcher.py tests/test_neral_reasoning_eval.py tests/test_neral_2022_gate.py
node --test tests/test_helios_history.js tests/test_arctus_history.js tests/test_issue195_assistant.js
node tests/test_issue195_assistant_browser.cjs
python -m pytest -q --tb=no
```

The browser test supports existing `HELIOS_TEST_PYTHON` and
`HELIOS_TEST_BROWSER` overrides. CI uses its installed Chromium.

## Product and model evidence

This fixes a product reliability defect and prevents inflated DEV scores.
It neither modifies Qwen weights nor improves trained Arctus350 weights.
Real-model answer quality, truncation frequency, throughput/VRAM and GPT-3.5
comparability remain unmeasured. Keep this PR Draft and stacked on #210;
do not merge, deploy or trigger training.
