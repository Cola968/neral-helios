# NERAL QLoRA pilot — isolated, review-only

This prepares NERAL, not a renamed Qwen release. The source base is Qwen/Qwen2.5-Coder-7B-Instruct, and the independently developed Arctus model family stays separate.

## What is included

- Ten small, code-grounded training examples for a pipeline smoke test. These do not demonstrate a quality improvement.
- A strict JSONL validator. Source rows contain system/user/assistant messages; the runner converts these into conversational prompt/completion records so completion-only loss is applied.
- A GPU runner with explicit opt-in. Default use only validates and prints a plan. The script is NOT a verified 12 GB GPU training run.
- PEFT adapter-only saving, not 16-bit base-model merging. Output outside the repository is required.

## Offline preflight (no CUDA, downloads or weight changes)

Run from the root of the stacked PR #219 checkout:

    python scripts/neral_qlora_pilot.py
    python -m pytest -q tests/test_neral_qlora_pilot.py

## Optional local training pilot (separate explicit approval required)

First create a separate, compatible GPU-training virtual environment. Do not replace the working inference venv or CPU environment. Pin and verify torch, transformers, trl, datasets, peft, accelerate and bitsandbytes, then run pip check. Unsloth is not required for this first PEFT pilot; adding it requires verifying Windows/CUDA compatibility separately.

Use the existing hash-verified, local, FLAT Qwen snapshot directory that contains helios-manifest.json. No remote model download is performed. Choose an output directory outside Git and any synced folder.

    python scripts/neral_qlora_pilot.py --model-dir 'C:\PATH\TO\VERIFIED\SNAPSHOT' --output-dir 'C:\PATH\OUTSIDE\REPO\neral-adapter-pilot' --steps 3 --max-length 512 --run-local-gpu

The pilot starts with LoRA r16, microbatch 1, gradient accumulation 4, 512 tokens and only 3 optimizer steps. These are cautious test settings, not an optimized full training recipe. Monitor VRAM and stop if memory errors occur. No specific VRAM ceiling has been verified. The script loads only local verified weights, hashes the dataset and writes a small adapter manifest.

## Not yet implemented

- Loading a PEFT adapter into Helios. PR #219 currently rejects adapters by design. Build a separate reviewed adapter import, compatibility check and rollback mechanism.
- Demonstrated model improvement, held-out evaluation, reproducible Windows CUDA training or release fitness.
- Auto-merge, cloud training, deployment or automatic paid jobs.

## Technical references

- docs/NERAL_QWEN25_GPU_SMOKE.md in PR #219
- https://huggingface.co/docs/trl/sft_trainer
- https://huggingface.co/docs/peft
