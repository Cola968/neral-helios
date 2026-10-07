# Arctus 350M — PC-off chat preparation

This branch prepares the existing `frontier350-v1` foundation checkpoint for a dedicated
chat post-training fork without changing the foundation lineage.

## What can be done while the RTX PC is off

- audit and refactor repository code;
- lock checkpoint/tokenizer compatibility contracts;
- build and unit-test assistant-only SFT formatting;
- prepare dataset validators and frozen chat evaluation;
- run CPU/static tests on GitHub-hosted or remote cloud workers;
- review licensing, provenance and benchmark separation;
- prepare the exact command/workflow that will use the local checkpoint later.

## What cannot be done while the PC is off

The current 350M checkpoint, tokenizer binary and training corpora intentionally remain local.
Therefore the actual weights cannot be loaded, updated or benchmarked on the RTX 4070 until
the self-hosted machine is online, unless those private assets are deliberately uploaded to a
separate cloud GPU service.

## Handoff policy

The chat fork must:
1. accept only the exact `frontier350-v1` architecture;
2. require the locked 16K BPE SHA-256;
3. start from an immutable foundation checkpoint with at least 1B processed tokens;
4. start a **fresh post-training optimizer** rather than pretending to resume missing/legacy optimizer moments;
5. supervise assistant tokens and EOS only; user/prompt tokens are context;
6. include base-LM replay and frozen generation gates to prevent catastrophic language loss;
7. keep the foundation checkpoint untouched.

The current lock is `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`.

## Next GPU-on stage

When the PC is online, the first run should be a bounded chat-recovery pilot from the best
immutable 350M checkpoint. It should measure SFT validation, held-out language loss, EOS,
repetition, instruction following, multi-turn state and throughput before any longer post-training.
