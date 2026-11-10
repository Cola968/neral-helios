# Arctus 24k BPE tok/s candidate

Goal: prepare a safer production configuration that can plausibly move Arctus from the current ~18k BPE tok/s production range toward 24k BPE tok/s on the project RTX 4070, without changing model weights, architecture, tokenizer, optimizer family, LR schedule, or effective tokens per optimizer update.

## What is measured vs targeted

Measured project references:

- 100M production median: ~18,463.72 BPE tok/s.
- Stage-5 winner: 20,585.4367 BPE tok/s with microbatch 8, grad accumulation 4, default compile, fullgraph.

Target:

- 24,000 BPE tok/s sustained production throughput.

The 24k number is not yet measured. No GPU run was performed while preparing this profile.

## Candidate v1

- sequence length: 512
- microbatch: 8
- gradient accumulation: 4
- tokens/update: 16,384, unchanged from MB4/GA8
- compile mode: default
- fullgraph: enabled
- numerical host readback: every 20 updates instead of every update
- log cadence: every 20 updates instead of every 10
- validation cadence: every 1,000 updates instead of every 500
- checkpoint cadence: every 1,000 updates instead of every 500

## Why this can be faster

1. MB8/GA4 halves the number of forward/backward microsteps per optimizer update while preserving the same effective token batch. This exact shape was the previous Stage-5 throughput winner.
2. Fullgraph removes more Python/graph-break overhead and was already part of the measured Stage-5 winner.
3. The current production command requests `--numerical-check-every 1`. In `train350_compiled.py`, each numerical check converts CUDA scalar tensors to Python floats. That introduces a device-to-host synchronization every optimizer update. Moving the safety readback to every 20 updates removes 95% of those host synchronization points while keeping finite-value checks frequent.
4. Logging itself explicitly synchronizes CUDA and writes JSONL. Halving log frequency reduces this overhead.
5. Validation and multi-GB checkpoint writes interrupt the steady training path. Halving their frequency reduces wall-clock overhead without changing optimization math.

## Guardrails

- Do not claim 24k until a real RTX 4070 benchmark measures it.
- Do not use `reduce-overhead`/CUDA-graph mode: prior Stage-6 attempts hit overwritten-output failures.
- Do not force Flash SDPA: the prior forced-Flash case was slower than default dispatch.
- Do not adopt the fused `model350_fast.py` execution layout for training without an optimizer-state conversion/equivalence gate.
- If MB8/GA4 fullgraph fails numerically or by memory on a future checkpoint, fall back to the existing MB4/GA8 default-compile production profile.
- The currently running 500M job is not modified or restarted by this preparation.

## Promotion gate for a future GPU test

A future short benchmark should promote this profile only if all of the following hold:

- sustained median >= 22.5k BPE tok/s and preferably >= 24k;
- no non-finite loss or gradients;
- no regression in fixed validation beyond ordinary sampling noise;
- no checkpoint/resume incompatibility;
- stable VRAM below the 12 GB RTX 4070 limit;
- compiled-vs-reference numerical gate passes.

Until then this remains `arctus-24k-candidate-v1`, not the production speed claim.
