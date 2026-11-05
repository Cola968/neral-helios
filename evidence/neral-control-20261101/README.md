# Shared control evidence — final verification 2 November 2026

Base: PR220 ef65d9c978144280754aee33e1fc101537550e31. New source files
only; original app/assistant/Qwen runtime/stores/core UI and training pipeline
were compared unchanged. No weights, private data, GPU inference, model generation,
training, paid job, merge or deployment was performed.

`cpu-tests.txt`: 448 selected tests passed; 1 native Windows test skipped.
Command: PYTHONDONTWRITEBYTECODE=1 HELIOS_DEVICE=cpu CUDA_VISIBLE_DEVICES='' 
/tmp/neral-control-check/bin/python -m pytest -p no:cacheprovider -q with:

- tests/neral_control (all new module/controller/provider/Flask/browser tests)
- tests/test_neral_qlora_pilot.py, test_neral_adapter_candidate.py,
  test_neral_blind_comparison.py, test_neral_project_sft.py
- tests/test_qwen_local_runtime.py, test_qwen25_gpu_smoke.py,
  test_helios_qwen25_store.py, test_arctus_flask.py
- tests/test_helios_runtime.py, test_helios_chat_experience.py,
  test_helios_workspace.py, test_helios_i1_ready.py, test_helios_install.py
- tests/test_issue195_assistant.py, test_neral_assistant.py,
  test_neral_assistant_code_tool.py, test_neral_dual_tool.py,
  test_neral_dual_tool_gate.py, test_neral_assistant_launcher.py,
  test_neral_reasoning_eval.py

Local network access was enabled for gateway/Flask/browser fixtures. Actual app
browser test uses HELIOS_PREPARATION_MODE=1 and temporary empty assets/state,
Chromium /usr/bin/chromium, Playwright, 360/390/768/1440 widths, keyboard submit,
math 125*48=6000, static code checking, profiles, errors and session bootstrap.
A labelled cooperative controller fixture tests real cancellation endpoint and
late-ack/final-result race; no model generates that cancellation response.
Temporary server/browser are closed. Native Windows file/junction/race testing
was NOT performed; the skipped test is a release blocker.

Additional checks: Node --check static/neral-control.js; Node --test existing
helios_history, arctus_history, issue195_assistant, qwen25_history: 4/4 passed.
compileall of new modules/tests/launcher, pip check and git diff --check passed.
The check venv uses Python3.12.14, torch2.8.0+cpu, pytest8.4.2, Flask3.1.3,
NumPy2.2.6 and waitress3.0.2. The inference/training environments were untouched.

RED/GREEN evidence: controller20 initially absent; malformed/repeated/truncated
calls then passed; cancelled usage and failed transport provenance separately
failed before their fixes. Wrapper version/weight status and strict metadata
positive/negative cases were tested before fixes. Additional regression caught
untrusted local role delimiters, tools-only startup that could inherit model-load
settings, and the POSIX-only UI availability gate before correction. Browser
caught late cancel acknowledgment overwriting terminal status before correction.
Win32 traversal mock tests were added to the interrupted implementation and are
contract tests, not a native security proof. No full green repository suite claim:
an earlier unfiltered attempt had missing-asset/socket/concurrent-edit failures;
this explicit final selected suite replaces intermediate extension failures.

`public-search-check.json`: actual free Wikipedia query, opt-in test only. It
records provider, real retrieved article URL/extract, API fetch URL, retrieval and
revision times. Not model knowledge or general search-quality evidence.

`recovery-operator-wrapper.txt` and `recovery-tests.txt`: isolated, tested operator
wrapper around unchanged pinned PR214 CAS client; not a product recovery service.
13 unittest executions (inherited scenario repeats included), including a
concurrent READY insertion refusal, passed before the authorized live operation.
Agent-tree inspection showed only current root; process inspection excluding its
ancestors showed no previous task executor. Guarded recovery changed only own key
neral-shared-control-221-v1, attempt1->2, from revision50 through the existing
claim/resolve/claim operations. Every snapshot/CAS retry refuses foreign READY or
expired RUNNING tasks or any changed foreign state. Other review/blocked tasks
were preserved. No token values are persisted here.

These tests establish system contracts. They do not establish learned NERAL
capability, a 100-task empirical comparison, a20% gain, long-term memory, real
Qwen tool-generation quality, Windows security or release readiness.
