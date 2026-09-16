# Neral NextGen V1 — isolated 50k development (2026-09-16)

**Status: experiment, NOT production.** Neral's original 349,825,152-parameter architecture and weights are preserved. This branch contains a synthetic RTX4070 compute probe, CPU evidence validator, local audited SFT pack builder and opt-in bounded SFT pilot; no new own-weight SFT run or verified quality gain has occurred. The prior V8.1 run ended in failure and was NOT restarted. No checkpoint has been overwritten, no GPU cloud charges or Hugging Face uploads have been initiated, and this PR stays draft.

## What 50k means

The user's revised primary goal is **>=50,000 accepted, end-to-end training tokens per wall-second**, sustained across three complete healthy stages of at least 5M accepted tokens each. Timing must include real corpus handling, transfers, checkpointing, validation and recovery. All stages and the aggregate must pass, with no rejected work, verified checkpoint and input provenance, unchanged hardware safety and no held-out quality regression. `admission.py` is a fail-closed *report-only* contract implementing these requirements; >=70k is a separate optional stretch. Its JSON input is not independently authenticated and it NEVER promotes a model.

**Synthetic compute-only rate is a different metric.** The code also measures whether an isolated microbenchmark can cross an intermediate 50k synthetic target, but this does NOT satisfy the accepted-token goal or change the production model. Synthetic loss changes are not quality measurements.

## Measured RTX4070 evidence

- [First GPU run](https://github.com/Cola968/neral-helios/actions/runs/35084101942): batch 4, sequence 512, four trainable upper blocks, eager 24,700 versus `torch.compile(default)` 38,129 synthetic tokens/s, 12 timed updates, no real checkpoint or data. See `GPU_PROBE_RESULTS_2026-09-16.md`.
- [Matched batch-8 run](https://github.com/Cola968/neral-helios/actions/runs/35084877377): same model hash, batch 8, sequence 512, four trainable blocks, 5 warmups, 20 timed steps: eager 32,626 versus compile 42,877 synthetic tokens/s. Peak compiled reserved VRAM 2.38 GiB and peak temperature 63 C. All steps and artifact upload succeeded; only a fixed synthetic batch was reused.
- [Guarded 50k shape sweep](https://github.com/Cola968/neral-helios/actions/runs/35085807321): new `probe.py` permits batches 12 and 16 for four trainable blocks only, max 8192 tokens/batch; measures batch12 compiled then matched batch16 eager and compiled, 5 warmups and 24 timed steps, explicitly checks temperature <=82 C and VRAM reserved <=9 GiB. It records compilation/setup+warmup time separately from timed steady-state throughput, slowest measured step and target flag. A failed candidate stops the workflow; evidence upload runs even on failure. Read actual run/artifact status before claiming a result.

The probe still excludes corpus, host-device transfer, replay, validation, checkpoint/recovery, exact long resume, NIS2, independent model generation and accepted-stage rates. It uses a fresh randomly initialized model and a fixed artificial token batch, so the loss is not meaningful for model quality. Batch size can improve *aggregate hardware utilization*, not reduce per-example computation or guarantee generalization. Compute-only rate cannot be compared numerically with a separate full production stage without matched boundaries.

## Code and checks

`probe.py`: explicit `--execute`, BF16, 349,825,152-parameter and RTX4070 checks, four/eight/twenty-four trainable block choices, tested larger shapes restricted to four blocks, fused AdamW, finite-loss and gradient checks, thermal/VRAM preflight and per-update guards, separate output and no checkpoint mutation. Its performance evidence v2 includes setup-inclusive and timed rates; v1 reports from earlier runs remain independently interpretable.

`test_probe_shape.py` tests oversize and train-block rejection without importing PyTorch. CI additionally syntax-checks the probe and tests the 50k accepted-stage admission boundary, SFT packing, data hashes and pilot loader. [GitHub CPU CI](https://github.com/Cola968/neral-helios/actions/runs/35085813631) passed 41 synthetic tests; this is not CUDA or model evaluation.

`SFT_PILOT.md` describes the unexecuted `train_sft_pilot.py`: 1–50 updates only, trusted SHA-pinned *own* base checkpoint and identical tokenizer, masked user-prefix supervised assistant response with EOS, local tiny held-out loss and isolated unpromoted output. Before any real SFT, independently inspect licenses, source provenance, near-duplicate/PII filters and held-out prompt contamination, then compare frozen free generation, EOS, math/code and full NIS2 with baseline. The 140-case historical diagnostic is not the entire NIS2.

## Hugging Face and other connections

The authenticated HF account was confirmed as `Cola968` without Pro; no HF Job, model upload or paid training was started. HF metadata for [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) lists Apache-2.0 and multilingual data, including German and English; metadata alone does not authorize unreviewed ingestion. This source remains a research candidate only, with no Hugging Face weights used. The Composio tool was unavailable (`FORBIDDEN` in this conversation); GitHub and HF were accessed directly, not through a bypass.

## Promotion boundary

Experimental speed alone never justifies a restart or merge. Require independently audited end-to-end accepted rate >=50k for all three stages and aggregate, identical data/budget/protocol, a safe and exactly resumable own-weight checkpoint, no validation/generation/math/code/safety regression, and separate sealed holdout before considering production. Exact-resume long-run NextGen training has NOT been implemented yet. Historical 60–70k aspirations are not measured achievements.

Technical references: https://docs.pytorch.org/docs/stable/generated/torch.compile.html ; https://huggingface.co/docs/transformers/main/perf_train_gpu_one ; https://huggingface.co/docs/trl/sft_trainer .
