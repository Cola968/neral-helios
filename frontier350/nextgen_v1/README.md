# Neral NextGen V1 — isolated 50k development (2026-09-16)

**DRAFT, experimental, not production.** Neral's own 349,825,152-parameter model and checkpoint weights are preserved. No own-weight SFT run, foreign weights, paid HF compute, corpus/checkpoint upload, merge, production restart or real quality gain. V8.1 ended with failure and has NOT been resumed; V8.2 frozen diagnosis completed independently.

## Precisely defined 50k goal

The primary requested goal is **at least 50,000 accepted end-to-end training tokens per wall-second**, sustained across each and aggregate of three complete healthy stages with >=5M accepted tokens/stage. Include actual data loading, H2D, forward/backward, optimizer, validation, checkpoints and recovery; reject any rejected tokens or safety/quality regressions. Require same own tokenizer, audited data order and budget, model, evaluation protocol and sealed holdout. `admission.py` implements a **fail-closed report-only** 50k gate; optional 70k stretch is independent. Neither synthetically invented test fixtures nor self-reported JSON are proof of real acceptance.

## Real hardware tests, but **synthetic**

All GPU rates below reflect the same fresh random Neral architecture, only 55,158,272 upper-four-block parameters trainable, repeated artificial tokens, RTX4070 BF16, fused AdamW, not a real training rate, learned quality or 24-block full-model benchmark. See linked artifacts for exact durations/VRAM/temperature and other boundaries.

- [Batch4 workflow](https://github.com/Cola968/neral-helios/actions/runs/35084101942): eager 24,700 / compile-default 38,129 synthetic tokens/s.
- [Matched batch8](https://github.com/Cola968/neral-helios/actions/runs/35084877377): eager 32,626 / compile-default 42,877.
- [Batch12/16 sweep](https://github.com/Cola968/neral-helios/actions/runs/35085807321): compiled batch12 44,160; batch16 eager 30,227 versus compiled **44,839** synthetic tokens/s; 3.59 GiB reserved and 63 C. Batch inflation showed diminishing returns; see `GPU_SHAPE_RESULTS_2026-09-16.md`.
- [Initial CUDA Graph A/B](https://github.com/Cola968/neral-helios/actions/runs/35086239657): default control passed, but reduce-overhead failed during capture/replay due globally cached RoPE cosine/sine tensor storage being overwritten. No run beyond failure; baseline artifact preserved.
- [RoPE prewarm recovery A/B](https://github.com/Cola968/neral-helios/actions/runs/35086701433): isolated probe precomputes the identical RoPE table **outside capture**, leaving `model350.py` and weights untouched. Both batch8/16 `reduce-overhead` runs passed and strict matched comparator tests passed. Yet default vs CUDA graphs = **43,002 vs 42,841** (batch8) and **44,815 vs 44,756** (batch16). CUDA graphs provided NO synthetic throughput improvement and neither reached 50k. See `GPU_CUDAGRAPH_RESULTS_2026-09-16.md` and six uploaded JSON reports. Setup and compilation times are separately reported and vary by cache state; do not treat the apparent setup reduction as proved speedup.

**Decision:** standard `torch.compile(default)` remains the controlled synthetic baseline. Do not enable reduce-overhead in production merely because a cache compatibility issue was fixed. No full-model, real-data, numerical parity, checkpoint exact-resume, accepted-stage 50k or quality proof. Historical V8.1 full training rate around 3.5k tokens/s must not be conflated with these four-block synthetic numbers.

## Implemented and tested

`probe.py`: explicit `--execute`, RTX4070/BF16 and architecture identity, guarded batch max 8192 tokens and large batch only with four trainable blocks, GPU idle preflight >=9000 MiB free, start <=78 C, run <=82 C, reserved <=9 GiB, finite loss/gradient. Separates steady-state compute-only rate from compile/setup-inclusive rate, gives slowest update and unpromotable flags. Optional reduce-overhead now prewarms RoPE outside capture, only on isolated research branch.

`compare_probes.py`: checks a matched two-report A/B for schema/classification/model SHA, GPU, trainable count, tokens, measured steps, shape, safety/VRAM limits, arithmetic and setup-inclusive consistency, and refuses false production status; comparison alone never asserts numerical parity or actual accepted training. `admission.py` separately checks 50k true-stage evidence and quality non-regression. Synthetic CPU unit tests cover shape policy, A/B manipulation, EOS-response packing, SHA/provenance checks and stage thresholds.

[CPU CI run](https://github.com/Cola968/neral-helios/actions/runs/35086473282): **48 tests passed**, syntax check completed; CPU contracts are not GPU capability tests. GPU recovery workflow also passed 11 isolated CPU shape/comparator checks and both GPU pairs.

`SFT_PILOT.md` documents `train_sft_pilot.py`, a **written but not executed** SHA-pinned, local own-weight SFT experiment (1–50 steps, tiny validation set, answer-only targets and EOS). Independent license, privacy, near-duplicate and benchmark-contamination screening required before real data; baseline must be compared on free generation, EOS, math/code, full NIS2 and held-out loss. The historical 140-case diagnostic is not full NIS2. Exact long-run resume for a NextGen foundation trainer has NOT been implemented.

## HF / Composio and next evidence boundary

Authenticated Hugging Face account `Cola968` was checked; `OpenAssistant/oasst2` metadata advertises Apache-2.0 and multilingual DE/EN, but it has not been ingested, independently vetted or used for weights. No HF Job or upload was launched. Composio returned `FORBIDDEN` for this conversation and was not used or bypassed. Official background: https://docs.pytorch.org/docs/stable/generated/torch.compile.html and https://huggingface.co/docs/transformers/torch_compile .

Performance follow-up should **profile actual bottlenecks** on a trusted isolated Neral checkpoint and realistic data accounting: synchronize only when required, check model forward/optimizer, data I/O, checkpointing/recovery, validate and compare true accepted wall rates and independent numerical/quality parity. Do not blindly grow batch or promise 50k when even the four-block synthetic ceiling is ~44.8k. The research PR stays unmerged until independent reproducible real-data evidence exists.
