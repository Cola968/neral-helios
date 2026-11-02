# NERAL Weight-Space Research – Inert Qwen2.5-Coder-7B Probe

Status: experimental, CPU-contract-tested, not model-quality-validated.
Tracking: GitHub issue #232. Based on QLoRA draft #220. This module is
not called by Helios and is not permission to start a GPU experiment.

## Research hypothesis and distinction from MIT CSAIL

The 2 November 2026 MIT CSAIL work on Weight-Space Programming reports
a targeted correction to a repetitive failure mode of Qwen3.5-0.8B.
That result is not evidence that Qwen2.5-Coder-7B contains the same
failure mode, layer locations, or correctable direction.

The code in scripts/neral_weightspace_probe.py exposes two independent
experiments, not an automatic model-improvement system:

1. Aggregate activations entering one Qwen2 down_proj projection during
   separately labelled failure and success cohorts. Never store raw
   activations, user messages, or secret project data in GitHub.
2. If an independently reviewed candidate direction exists, apply a
   bounded, in-memory forward-output correction to a frozen eval-only
   instance of the model. Uninstall the hook when the context exits.

For a bias-free linear projection, with unit direction u, the edit is

    y_new(x) = y_original(x) - strength * (x dot u) * y_original(u)

and is algebraically equivalent to the *linear-layer* rank-one matrix
expression W_new = W - strength * (W @ u) @ u.T. The implementation does
not write a new weight tensor or dequantize/requantize saved NF4 weights.
It is neither a reproduction of the published MIT edit nor a general
causal identification method.

## Bounds and integration contracts

- Only explicit Qwen2/Qwen2.5 layer paths ending in mlp.down_proj are
  accepted, including a narrow PEFT-wrapped form.
- Layer input/output feature sizes come from declared projection metadata,
  cross-checked against the original quantization shape. Packed storage
  dimensions must not be treated as logical matrix dimensions.
- Corrupt/missing quantization metadata, biased projections, nonfinite
  vectors, unexpected input shapes, unfrozen training parameters, a model
  in training mode or existing layer hooks cause refusal.
- Edit strength must be greater than zero and at most 0.10. A separate
  experimental_opt_in=True is mandatory on the edit call.
- Activation capture defaults to 512 tokens, hard cap 4096 tokens. Only
  CPU float32 sums and counts are retained.
- An existing LoRA adapter may alter the activation distribution; a
  direction fitted for base Qwen may not apply to base plus LoRA.
- No shared concurrent serving instance is safe for a temporary hook.
- The caller must ensure no training, downloads, file writes or arbitrary
  execution is performed in the same experimental invocation.

## Baseline loop audit (read-only)

scripts/neral_weightspace_loop_audit.py accepts bounded, locally stored
JSONL records. Each record has EXACTLY these fields:

    task_id, base_model_id, base_revision, prompt_sha256,
    tokenizer_sha256, generation_config_sha256, adapter_sha256,
    max_new_tokens, finish_reason, output_token_ids

The model must be Qwen/Qwen2.5-Coder-7B-Instruct with a 40-hex revision.
For a pure base-model baseline adapter_sha256 must be null. EOS token ID
is 151645. finish_reason is eos, length, cancelled, or failed; output
token count and termination metadata must match. Bounded rows (up to
2000, 2 MiB total) must use one common revision, tokenizer, generation
configuration and output budget, with unique task and prompt digests.

Run locally, with no GPU or model loaded:

    python scripts/neral_weightspace_loop_audit.py LOCAL_BASELINE.jsonl

Only aggregates are printed: completion categories, the proportion that
exhausted the output budget, and repeated four-gram fractions.
A tentative review flag currently requires a completed *length-limited*
answer of at least 32 tokens and a repeated four-gram fraction >= 0.35.
A batch signal requires >= 20 eligible cases with >= 3 suspects and
>= 15% of eligible cases flagged. These are uncalibrated triage
thresholds, not tests of statistical significance or model intelligence.

Repetitive Python/JSON boilerplate and an inadequate token budget can
create false positives. A loop flag means review the task and budget,
not edit weights. No raw prompt or output is emitted by the audit.
The input token IDs may themselves encode private data; keep source
JSONL local and private. The audit does not independently verify that
records were produced by the claimed model.

## Tests and outstanding validation

Run focused CPU tests from the repository root:

    python -m unittest discover -s tests -p 'test_weightspace*.py' -v

Local V4 source suite: 26 CPU tests, all passing. It includes synthetic
packed-four-bit and PEFT stand-ins, not real bitsandbytes CUDA kernels.
No true Qwen2.5-Coder-7B NF4 inference, no live adapter, no real model
baseline, no intervention against unseen tasks, and no quality gain have
been measured. Preserve this distinction in every status update.

Before even considering an experiment with the RTX 4070, require:
a verified local snapshot and current package/CUDA compatibility;
real baseline outputs including accuracy, code syntax, EOS/truncation
and repetition; independent diagnosis of whether a genuine pathological
loop exists; a separately approved, read-only GPU experiment; a
registered direction fitted only on development data; rollback checks;
and blind comparison against untouched held-out tasks. Regression in
coding or instruction following rejects the candidate.

Release or promotion is forbidden without independent quality evidence.
No repository model weights, HF snapshots, checkpoints, or private
activations should be uploaded. No automatic merging, training or
model activation follows from this research module.
