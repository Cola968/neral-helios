# Evidence scope — 1 November 2026

This session is Linux/Python 3.12.14, without an attached RTX 4070 or Windows host.
The inference environment from PR #219 was not accessed or changed. No training,
optimizer step, paid job, model download, merge or deployment was performed.

- `linux-api-check.json`: actual pinned imports, tiny random CPU Qwen2/PEFT and
  SFTTrainer construction, collator completion/EOS/padding assertions, temporary
  safetensors adapter save/load roundtrip. No trainer.train, forward or backward.
  PEFT emitted its expected warning about a random in-memory base lacking a
  config path; this fixture is not the real local Qwen snapshot.
- `linux-freeze.txt`: actual isolated venv freeze, **not** a Windows lock file.
- `windows-wheel-audit.json`: actual downloaded Windows wheels/hashes plus torch
  cu128/cp312 Windows URL/hash from the official index. Availability only.
- `cpu-tests.txt`: 207 passed, including 28 pilot/candidate/comparison tests and
  existing Qwen, Helios, Arctus and assistant contracts.

A first regression invocation inside the default sandbox returned 203 passes,
one failure and three setup errors: socket creation raised PermissionError in
four existing local HTTP gateway cases. The unchanged suite passed with local
network access enabled (207/207). No gateway code was changed or test skipped.
PowerShell is unavailable here, so the new Windows setup script remains unexecuted.

Reproduction:

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/neral-training-check/bin/python scripts/neral_training_preflight.py
/tmp/neral-training-check/bin/python -m pip check
python scripts/neral_qlora_pilot.py
```

The regression command uses `HELIOS_DEVICE=cpu CUDA_VISIBLE_DEVICES=''` and
`python -m pytest -q` with these files:

```text
tests/test_neral_qlora_pilot.py tests/test_neral_adapter_candidate.py
tests/test_neral_blind_comparison.py tests/test_qwen_local_runtime.py
tests/test_qwen25_gpu_smoke.py tests/test_helios_qwen25_store.py
tests/test_arctus_flask.py tests/test_helios_runtime.py
tests/test_helios_chat_experience.py tests/test_helios_workspace.py
tests/test_helios_i1_ready.py tests/test_helios_install.py
tests/test_issue195_assistant.py tests/test_neral_assistant.py
tests/test_neral_assistant_code_tool.py tests/test_neral_dual_tool.py
tests/test_neral_dual_tool_gate.py tests/test_neral_assistant_launcher.py
tests/test_neral_reasoning_eval.py
```

Compile checks and `git diff --check` passed. Unchanged product paths were compared
against PR #219's `a27ff73a154535ed93e2cbe0bb373e04d7916e86`.

| Requested measurement | Actual result |
| --- | --- |
| Windows pip check / CUDA probes | Not run; host unavailable |
| NF4 backward / paged optimizer on RTX 4070 | Not run; training not authorized |
| Training peak VRAM / loss / speed | Unmeasured; report instrumentation prepared |
| Real trained adapter load / rollback on CUDA | Not run; no trained artifact |
| Independent baseline/adapter quality scores | Not collected; holdout and blind export prepared |

A review manifest binds ten source-reviewed data rows to source hashes. It records
this coding agent's audit, not independent human approval. Held-out prompts have
no exact training-prompt overlap. The comparison needs separately collected real
responses and an independent reviewer; no quality improvement is asserted.
