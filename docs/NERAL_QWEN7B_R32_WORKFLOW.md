# NERAL Qwen2.5-Coder-7B Rank32: bounded Windows GitHub Actions training proposal

This is a **review-only** workflow until explicitly merged by the repository owner. Merely opening a PR cannot make a new `workflow_dispatch` workflow runnable: GitHub requires the workflow to exist on the default branch. No automatic merge is enabled. Before merging, review the runner trust boundary and workflow-permission policy; `.github` scope is excluded by the existing shared queue.

The workflow checks out its own helper from the trusted `main` and the **exact** reviewed trainer/data snapshot from PR #234 (`a082738f9145722dec443141e5d7801a3ead93af`). It hashes the QLoRA trainer against `f4b0f72440ce40653ca0ce2b171bec6721505e18`, generates a separate rank32 copy, bundles 86 reviewed examples, and verifies the local flat `helios-manifest.json` and all model weight hashes before any deserialization. It never downloads the base model.

`workflow_dispatch` modes: `preflight` or `train-three-steps`. Training requires the exact `authorize` phrase `EXACTLY_THREE_OPTIMIZER_STEPS_R32`. The `install_pinned_environment` input is false by default; when true, only a *new* external Windows Python virtual environment and pinned dependencies are installed (no model download). This is not a silent permission to buy services. The workflow holds a single GitHub Actions concurrency group and samples GPU load twice. It refuses less than 10,000 MiB free VRAM or GPU utilization greater than 15%. It cannot reliably detect all **non-GitHub** local GPU processes; the user must avoid concurrent local training.

Local model selection is bounded to the configured `NERAL_QWEN25_HF_SNAPSHOT`, `NERAL_ASSETS_ROOT/installed_models/hf-qwen25-coder-7b`, or three documented known local directories. It deliberately **does not scan all of C:\\**, trust random `.safetensors`, or construct a manifest for an unverified HF cache. Missing/multiple sealed snapshots => fail closed, with no training. Existing configured local model and virtual-env directories are not overwritten.

The existing `scripts/neral_qlora_pilot.py` uses BF16/NF4, PEFT, completion-only loss, gradient checkpointing, microbatch 1, gradient accumulation 4, and `SFTConfig(max_steps=3, optim='paged_adamw_8bit')`. The separately generated rank32 trainer checks **80,740,352 trainable LoRA parameters** prior to optimizer steps and now records per-step optimizer durations. The original HF model weights are frozen and not saved.

Output adapter and full local reports remain under the runner user's `NERAL-QLoRA-Research/actions-r32-*` directory outside the repo. Only scalar JSON (loss per optimizer step, duration, three-step count, peak GPU bytes and SHA-256 of the **adapter** file) is allowlisted for upload. Artifact uploads MUST NOT include safetensors, private prompts, model weights, tokenizer files or datasets.

**Limits:** Nothing here proves the rank32 pilot fits 12 GB VRAM or that 3 optimizer steps improve model quality. A successful pilot proves training plumbing and creates a sealed candidate, not a new independent model. The proposed workflow must be reviewed and merged before it can be dispatched; existing task-queue restrictions on hidden paths require an explicit policy review. No existing PRs/branches or base model weights are modified by the proposal.

## Audit corrections prepared on 2026-11-04 (not committed to GitHub)

- The pinned `neral_adapter_candidate.py` rejects `r=32` in its
  `validate_adapter()` function. The updated `neral_rank32_copy.py` verifies
  the exact trainer **and validator** Git-blob SHA-1 values before any generated
  code is trusted, then creates a separate `neral_adapter_candidate_r32.py`.
  The original r16 validator remains unchanged. Both generated Python files
  must be present in the pinned checkout during the pilot. Validate against
  exact frozen source, not a fabricated/mutable external dependency.
- Windows GPU utilization is sampled three times; *every* sample must show
  at least 10,000 MiB free and at most one of the samples may exceed 15%
  utilization. This is a heuristic, not exclusive ownership of the GPU.
- The runner prints the selected manifest path and isolated Python executable
  for troubleshooting; these paths are log data, not secret tokens.
- **New durable one-shot protection:** the first explicitly authorized `train-three-steps`
  after a successful preflight atomically creates
  `NERAL-QLoRA-Research/.neral-r32-three-step-one-shot.lock` using Python's
  `os.O_CREAT | os.O_EXCL`. Any later training dispatch is refused, even with
  the same authorization phrase and a different GitHub run ID. A crash still
  consumes this authorization to prevent unreviewed automatic retries.
  The `preflight` mode does not consume it. A human operator must inspect any
  interrupted run before manually authorizing *another* pilot. There is no
  automatic delete/reset path.

Local evidence for this prepared revision: 19/19 stdlib/CPU tests passed,
`compileall` passed, actual pinned GitHub trainer and validator Git blobs
matched and the generated trainer/validator both passed Python AST parsing.
None of this certifies Windows PowerShell behavior, HF model availability,
training, GPU peak memory, loss or model quality. The new workflow still needs
explicit repository review and controlled registration on the default branch.
