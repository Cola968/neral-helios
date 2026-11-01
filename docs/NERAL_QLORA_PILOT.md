# NERAL QLoRA pilot — isolated and reviewable

NERAL remains the product name, Helios the application, and
`Qwen/Qwen2.5-Coder-7B-Instruct` the disclosed base. Arctus stays separate.
This draft is stacked on PR #219, not main. Target date: 10 November 2026.

## Verified on 1 November 2026

- New `/tmp/neral-training-check` Linux/Python 3.12.14 venv, torch 2.8.0+cpu,
  Transformers 4.57.1, TRL 0.24.0, PEFT 0.17.1, datasets 4.3.0,
  accelerate 1.11.0, bitsandbytes 0.48.2, safetensors 0.6.2.
- `pip check`, actual imports, tiny randomly initialized CPU Qwen2/PEFT/SFTTrainer
  construction, completion/padding/EOS masks and safetensors adapter save/load
  round trip. No forward/backward training or optimizer step. Temporary random
  fixtures are deleted; these are not trained NERAL artifacts.
- Windows Python 3.12 wheels for bitsandbytes, safetensors and tokenizers downloaded
  and hashed; official torch 2.8.0+cu128 Windows wheel URL/hash found in its index.
  Wheel availability is **not** Windows execution, driver compatibility, NF4
  backward, paged optimizer or 12 GB training evidence.
- Evidence: `evidence/neral-qlora-20261101/`. Linux freeze records transitive
  dependencies; it is not a Windows lock file. Core training packages are pinned
  in `requirements-neral-training.txt` and checked before training.

The working inference venv, CUDA installation, `requirements-qwen25-local.txt`,
`app.py`, model catalog and `qwen_local_runtime.py` are unchanged. No RTX 4070 is
attached to this Linux session. PR #219's successful Windows inference runs do not
prove training compatibility. Unsloth is not required, installed or certified;
adding it would require a separate version/API/kernel check.

## Audit and corrections

The original pilot's seven offline tests passed. They did not exercise TRL or
validate factual content. The following gaps are addressed:

- A format check is now named as such. Training additionally requires a matching
  reviewed dataset digest and source-file hashes. The ten existing responses were
  checked against `app.py`, `chat_store.py`, `code_check.py`, the Qwen runtime and
  model catalog. The review file records evidence per row, including the rows
  expressing response/evaluation policy rather than implementation facts.
  Hashes detect drift; they cannot prove semantic truth. Changes need fresh review.
- Tokenization uses the same audited Qwen chat template as inference. Prompt
  tokens are masked, completion and EOS retained. Overlong rows and inconsistent
  boundaries fail before the 7B model loads, rather than silently truncating labels.
- Exact Qwen2 classes, local-only safetensors and SDPA match the existing bridge.
  NF4 loading is checked; gradient checkpointing explicitly uses non-reentrant
  mode. Source snapshot/output paths must be separate. No base merge is performed.
- Trainer logs, final metrics and allocator/global-device memory samples are saved
  even if model loading/training/saving fails. Non-finite logged metrics abort.
- Adapter bundles bind version, base revision, snapshot integrity, dataset digest
  and safetensors/config hashes. Base tokenizer is reused; no imported adapter
  tokenizer or Jinja template is executed.

This is an audit of code and CPU interfaces, not an assertion that every possible
GPU failure has been eliminated. Dataset quality/size is sufficient for a pipeline
smoke test, not a general assistant or demonstrated independent improvement.

## Prepare Windows without touching inference

Use a new, unsynced directory outside Git and the existing inference venv:

```powershell
# Setup installs only into the NEW venv; refuses an existing target.
powershell -File scripts/setup_neral_training.ps1 -VenvDir C:\NERAL\venvs\qlora-pilot-01
$trainPython = 'C:\NERAL\venvs\qlora-pilot-01\Scripts\python.exe'
& $trainPython -m pip check
& $trainPython scripts/neral_training_preflight.py --check-cuda |
    Tee-Object C:\NERAL\windows-training-preflight.json
& $trainPython -m pip freeze | Set-Content C:\NERAL\windows-training-freeze.txt
nvidia-smi | Set-Content C:\NERAL\windows-nvidia-smi.txt
```

The CUDA preflight executes only tiny BF16 matmul and NF4 quantization/dequantization
and Linear4bit forward probes, no model download, backward or training. It requires
CUDA 12.8 torch and RTX 4070/BF16/at least 11 GiB total memory. Installed NVIDIA
driver must support that wheel; no global toolkit or driver is changed by setup.
PowerShell setup and CUDA probes have **not** been executed on Windows here.
A resolved Windows freeze and GPU preflight are required before calling these pins
Windows-validated. Stop and report errors; do not upgrade the inference environment.

## Offline checks

```text
python scripts/neral_qlora_pilot.py
python -m pytest -q tests/test_neral_qlora_pilot.py tests/test_neral_adapter_candidate.py tests/test_neral_blind_comparison.py
```

The default command performs format + reviewed-corpus checks, imports no training
libraries and writes no adapter. `--dataset` may be used for format inspection;
GPU training additionally requires its matching `--review-manifest`.

## Prepared GPU experiment — separate explicit approval required

Do not execute this command until training is separately authorized. First unload
Helios inference and close other GPU workloads. Use the previously verified flat,
read-only-by-policy local Qwen snapshot with `helios-manifest.json`, outside sync
folders. A fresh output directory per version is required.

```powershell
& $trainPython scripts/neral_qlora_pilot.py `
  --model-dir C:\NERAL\models\qwen25-verified `
  --output-dir C:\NERAL\adapters\0.1.0-pilot.1 `
  --adapter-version 0.1.0-pilot.1 --steps 3 --max-length 512 --run-local-gpu
```

Recipe: NF4/double quantization/BF16, SDPA, LoRA r16/alpha16/dropout0 on seven
projection types, microbatch 1, accumulation 4, non-reentrant checkpointing,
paged AdamW 8-bit, LR 2e-4, seed42, at most three initial optimizer steps.
If 512 tokens is insufficient for a reviewed row, review the length report and
memory budget before choosing a larger supported length; never silently shorten.

`training-report.json` records loss/gradient norm/LR logs, Trainer train runtime,
samples/sec and steps/sec, elapsed time including load/save, trainable parameters,
package/CUDA/device identity, peak allocated/reserved bytes and 100 ms sampled
whole-device usage. Device samples include other processes and can miss peaks;
allocator peaks exclude driver/other-process allocations. No 12 GB fit is promised.
Errors during early preflight/tokenization are terminal errors before output
creation; model load/train/save errors are recorded with failed status. Keep the
console log too. Failed directories are retained for diagnosis, never reused.

**No GPU training has run. VRAM, loss and training speed are unmeasured.**
OOM/non-finite loss/kernel/optimizer failures must be recorded before changing one
setting at a time. Do not auto-retry, merge adapters into base weights or upload
models/data. Three steps validate plumbing, not quality.

## Staged adapter integration and rollback

`neral_adapter_candidate.py` is an explicit experimental entrypoint, **not wired
into Helios**. It accepts only a flat, hash-checked standard LoRA recipe with exact
base revision/integrity. Every training artifact starts `release_validated=false`.
The manifest is an integrity record, not a signature or proof of quality; trusted
local storage and reviewed promotion remain necessary.

A caller must explicitly opt into an unvalidated pilot. The candidate uses the
verified base and local PEFT safetensors, freezes gradients, changes `runtime_id`
with adapter version/hash to prevent stale-context reuse, and returns adapter
status. If PEFT mutates the model then fails, it discards/unloads that model and
reloads a fresh base; failures of the fresh base are propagated. Pre-load integrity
errors reject the selection before loading. No silent promotion/active pointer.

Offline tests cover wrong base, tampering, links, extra code/pickle files,
unsupported config, version immutability, context identity and partial-load
rollback. Actual 7B NF4 adapter inference, cleanup/VRAM, streaming, cancellation,
UI provenance and rollback under CUDA still require approval and hardware checks.
Only after GPU smoke and independent evaluation should a follow-up connect this
entrypoint under Helios' existing generation lock. UI must show NERAL + adapter
version + Qwen base and expose base-only/prior-version selection. Existing native
and base-only paths remain available. No auto-activation of newly trained output.

## Independent comparison prepared

`datasets/neral_qlora_holdout.jsonl` freezes eight separate prompts/rubrics for
project correctness, uncertainty, product/base disclosure, coding, reasoning and
safety. No exact training prompt overlap; some concepts deliberately overlap to
measure transfer, so this small set is not a broad generalization benchmark.
Do not train/tune on these items after freezing them. An independent human reviewer
must score responses; the same coding agent producing data is not an independent
judge of model improvement.

Collect base-only and candidate responses in fresh separate processes with the
same verified base, versions, system prompt, audited template, NF4/BF16 settings,
greedy decoding (`do_sample=false`) and 256-token budget. Never execute generated
code automatically. Record all failures; do not remove difficult tasks.
Input response JSONL: `{"id":"identity-01","text":"..."}` for all eight IDs.
Provide one metadata JSON per run with matching `base_model`, `base_revision`,
`base_snapshot_integrity`, `holdout_sha256`, `system_prompt`,
`chat_template_sha256`, `max_new_tokens`, `do_sample`, `quantization`, `packages`;
`adapter_sha256` must be null for baseline and the candidate manifest digest for
adapter. Report per-case latency/output tokens and VRAM separately as well.

```text
python scripts/neral_blind_comparison.py --base base.jsonl --adapter adapter.jsonl --base-meta base-meta.json --adapter-meta adapter-meta.json --out NEW-review-dir
```

The offline exporter rejects unequal settings, duplicate/missing cases or a
changed holdout and shuffles A/B per item. Give only `reviewer.jsonl` to the reviewer;
retain `private-key.json` separately. Score each response 0=incorrect, 1=partial,
2=correct using the frozen rubric, note unsupported claims and regressions, then
unblind. Require no new identity/privacy/tool-fabrication failures and no drop in
coding/general tasks before considering activation. Eight cases only provide
pilot evidence; expand with a new unseen external set before any quality release.
No responses, scores, improvement or comparison result are claimed yet.

## Remaining gates to 10 November

1. 2–3 November: execute Windows setup, `pip check`, CUDA probes; preserve freeze,
   driver and logs. Block training if any fail.
2. 3–4 November: obtain separate authorization for one 1–3-step local GPU smoke;
   inspect memory, loss, speed, error report and saved artifact integrity.
3. 4–7 November: expand reviewed training data if the smoke passes; obtain fresh
   training authorization for any longer run. Keep held-out data separate.
4. 7–9 November: authorize adapter inference test; verify the candidate under CUDA,
   rollback and blind independent comparison; then review minimal app/UI wiring.
5. 10 November: release readiness review based on recorded evidence. No automatic
   merge, deployment, paid job or promise of improved quality by a deadline.
