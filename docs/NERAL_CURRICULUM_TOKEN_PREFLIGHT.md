# NERAL 1,000-Example QLoRA Tokenizer Preflight

Purpose: turn the 800-train, 100-validation, 100-test synthetic NERAL
curriculum into a precisely measured **local training candidate**.

This phase is a read-only proof. It does NOT train a new adapter, modify
the current LoRA, or claim a quality improvement.

## Review gates

1. Pin the four exact Git blobs comprising the training, validation and
   test JSONL files and the dataset manifest.
2. Run the existing deterministic 1,000-row numerical, SQL/code-structure
   and distinct-prompt validator; check that data remains review-only.
3. Load only the already installed local Qwen2.5-Coder-7B tokenizer and
   fixed chat template from the pinned original trainer.
4. For **every** example in all three splits: encode the system/user prefix
   with the generation marker, encode the full system/user/assistant turn,
   assert identical prefix and nonempty assistant-only completion, assert
   no silent truncation at the reviewed 512-token limit.
5. Report only split counts, minimum/maximum token lengths, source SHA-256,
   and the verified local Qwen revision.

The CPU unittest simulates a tokenizer; it must never be presented as a
real Qwen result. The manual Windows workflow performs actual model
snapshot and tokenizer integrity checks before asserting it is verified.

## Running

GitHub Actions workflow: NERAL 1000 synthetic rows Qwen tokenizer
preflight (no training). Select the main branch and manually enter:

RUN_NERAL_1000_TOKEN_PREFLIGHT

The manual action uses the existing self-hosted Windows runner and
local Qwen snapshot; no downloads, cloud inference, optimizer, adapter
write or private prompt upload. If the runner is offline, the job cannot
succeed. Existing source paths, trainer snapshots and QLoRA weights
are not overwritten.

Only a guarded summary.json artifact may leave the Windows runner.

## What passing does not prove

The source curriculum is synthetic and parameterized. Even unique
prompts can be near-duplicates of each other. The source review is
algorithmic, not independent human fact checking. Neither the
tokenizer check nor a lower training loss establishes general
intelligence or a release-quality adapter.

Before any new rank16/rank32 QLoRA experiment, establish a frozen
new adapter output path, independent coding/tool/core holdouts,
a token/VRAM budget, regularization and a stop-on-regression
comparison with unchanged Qwen foundation and existing NERAL adapter.
No existing rank32 three-step safety gate is disabled.
