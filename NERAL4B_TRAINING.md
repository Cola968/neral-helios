# NERAL Assistant 4B · Own adapter, not a new foundation model

**Stage (2026-10-27): code and CPU-only contract tests. No model weights have been
trained or uploaded. This is a candidate recipe and has NOT passed an actual
Unsloth/Qwen3.5-4B GPU compatibility or quality test.**

## Goal and separation

Build NERAL's first reviewable *adapted* language model: a LoRA adapter trained
over `Qwen/Qwen3.5-4B` (Qwen, Apache-2.0), while preserving the separate
self-pretrained Arctus research line. This work **does not** replace the
application's 9B Ollama baseline in PR #129; do not merge/release automatically.

For the existing NERAL 9B application integration: [PR #129](https://github.com/Cola968/neral-helios/pull/129).

Architecture facts confirmed from Hugging Face: Qwen3.5-4B has ~4.66 billion
parameters; license Apache-2.0; Qwen's released `Qwen3.5-4B` is itself a
post-trained model derived from `Qwen3.5-4B-Base`.
Model: https://huggingface.co/Qwen/Qwen3.5-4B

## Reviewable data preparation, even with the PC turned off

Inputs must be **two local, reviewed JSONL files**, `train.jsonl` and
`val.jsonl`, under a directory you choose. Each row must have exactly:

```json
{
  "id": "approved-example-0001",
  "prompt": "Erkläre kurz, was ein KI-Modell ist.",
  "answer": "Ein KI-Modell ist eine durch Training angepasste mathematische Funktion.",
  "source": "OpenAssistant/oasst2",
  "license": "Apache-2.0",
  "revision": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
  "reviewed": true,
  "language": "de",
  "domain": "general"
}
```

This is a **format illustration, not a downloaded, verified OASST2 training
example**. Do **not** mark it as a real OASST2 row. Real OASST2 data must match
the pinned project-reviewed archive. If using genuinely original, manually
reviewed NERAL material, use `source=neral-original`,
`license=CC-BY-4.0`, `revision=neral-reviewed-v1` **only after authorship,
usage rights and license have been reviewed**. A string label alone does not
prove ownership, the review or the correctness of the data.

The tool rejects arbitrary sources, unreviewed samples, unrecognized licenses,
unfrozen source revisions, duplicates, overlapping normalized prompts across
train and validation, malformed fields, oversized input, and symlink source
files. Public benchmarks such as GSM8K or MBPP must never be used as train data.
The existing [HF_DATA_POLICY.md](HF_DATA_POLICY.md) remains authoritative.

Prepare a local dataset package (requires Python 3.12; **no dependencies**):

```sh
python neral4b_training.py --data-dir /path/to/reviewed_sft
```

It only writes `local_state/neral4b/prepared/{train,val}.jsonl` and
`manifest.json` and prints its data fingerprint. The prepared records are
conversational **prompt-completion** examples: system + user prompt, followed
by assistant completion. Training loss is intended to affect completions only,
not the prompt. Preparation works offline and cannot use cloud GPU services.

**A trainable dataset is not shipped in this PR.** The existing historical
NERAL/OASST2 preparation scripts are separate; this PR does not copy or
relicense their outputs without verifying source lineage. The dataset
preflight requires at least 100 training examples and 20 separate validation
examples before weight-changing training is permitted. This is only a
minimum engineering gate, *not* an assertion of sufficient data quality.

## Candidate GPU recipe (only by explicit approval)

Unsloth's official Qwen3.5 text fine-tuning guide supports Qwen3.5 and warns
against QLoRA 4-bit quantization for this model family because of quality
differences. This experimental recipe therefore prefers **BF16 LoRA** on
Qwen3.5-4B (rank 16, 1024-token context, batch 1, accumulation 8, capped 80
optimizer steps). Source:
https://unsloth.ai/docs/models/qwen3.5/fine-tune

> **Do not run this yet without explicit user approval, dataset admission, a
> compatible Linux CUDA/BF16 GPU, spending cap and a short GPU smoke test.**
> Dataset and CPU validation alone do not validate the GPU stack.

Under a *user-approved* GPU session, install appropriate compatible versions
of PyTorch/CUDA, Unsloth, Transformers v5, TRL, PEFT, datasets and vision
dependencies as documented upstream, and verify the exact versions before
any training. Then *and only then* explicitly opt into both flags:

```sh
export NERAL_4B_TRAIN_APPROVED=YES
python neral4b_training.py --data-dir /path/to/reviewed_sft --execute
```

The `--execute` flag **and** the environment variable are both mandatory.
The recipe saves a local adapter under
`local_state/neral4b/experiment/adapter`; it does **not** automatically push
to Hugging Face, merge weights, upload personal data, or change the 9B runtime.
GPU infrastructure, cloud billing and job scheduling are outside this script.

**Important limitations:** exact GPU memory and runtime are unverified; BF16
support is required; the Unsloth/TRL API can change and needs a real smoke test.
It is possible that the text-only trainer needs further adaptation for the
Qwen3.5 multimodal architecture. A real trained adapter and train/validation
loss curves are prerequisites to claiming a model quality improvement.

## Release gates and 12-hour budget

1. Review/check rights and hashes of a clean NERAL dataset from approved
   sources, no benchmark contamination.
2. Run the offline preflight with a fixed source manifest.
3. After permission, conduct a **small bounded** GPU loader + one-step
   smoke test; verify memory/actual model type and token masking.
4. Compare raw Qwen3.5-4B against trained adapter with **identical frozen,
   unseen evaluation prompts**, using both functional and safety tests.
5. Document quality/regressions, model provenance, license notices and
   tested adapter loading. Only then consider public release.

A 12-hour rental is a **maximum budget**, not a reason to spend all of it:
begin with a short checkpointed experiment if authorized. No claims of
Giotto-level performance; keep a stronger Qwen3.5-9B serving baseline
independently.

Offline tests:
```sh
python -m pip install pytest
python -m pytest -q tests/test_neral4b_training.py
```

CI: `.github/workflows/neral4b-preflight.yml`. CI runs with GPU explicitly
disabled and Hugging Face/model download access disabled. No 4B model files
are fetched in CI.
