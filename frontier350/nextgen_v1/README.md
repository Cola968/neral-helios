# Neral NextGen V1 — 44k stability and model-quality research (2026-09-16)

**Draft and experimental; not production.** The target was reduced at the user's request from 50k to **44,000 accepted end-to-end tokens/s**. This is a *new acceptance threshold*, not a claim of measured real training throughput. Preserve Neral's own 349,825,152-parameter model and existing checkpoints. V8.1 ended with failure, has not been resumed, and no new training/own-weight SFT or quality improvement is claimed. Do not merge or overwrite checkpoints based on synthetic speed.

## Revised objective: stable real training and quality, not peak compute speed

`admission.py` is a report-only, fail-closed provisional evidence check. It requires >=44,000 **quality-accepted training tokens per total elapsed wall-second** in EACH of three consecutive, independently verified healthy stages of at least 5 million accepted tokens and in their aggregate. Include data I/O, host-to-device transfer, forward/backward, optimizer, validation, checkpoint and recovery time; rejected tokens are never progress. Model identity, own weights and tokenizer, fixed real-data order and budget, RTX4070 hardware and thermal/VRAM safety and evaluation protocols must match. No validation loss, repetition or math/code/science/language development-category regression; post-training requires better held-out answer correctness and EOS behaviour. Independently verify full NIS2, additional frozen generation sets, exact checkpoint resume and provenance before any production promotion. Contract fixture success is NOT measured evidence; JSON can be forged and the validator does not independently authenticate it.

Do not restart production or launch unattended multi-hour training. The next real-data, isolated own-checkpoint trial must first demonstrate exact recovery, GPU headroom and reproducibility; then run the three healthy stages and inspect primary source artifacts. If any one stage is slow, unhealthy, rejected or regresses quality, do not promote it even if its aggregate speed looks good. Details: `STABILITY_44K_PLAN.md`.

## Historical *synthetic* GPU results (not accepted training throughput)

RTX4070 BF16; fresh random Neral architecture, only the upper four blocks trainable (55,158,272 parameters), same repeated artificial token batch, no real corpus or checkpoint, no accepted-stage accounting or NIS2. These cannot establish the new 44k real objective.

- [Batch 4](https://github.com/Cola968/neral-helios/actions/runs/35084101942): eager 24,700; `torch.compile(default)` 38,129 synthetic tokens/s.
- [Batch 8](https://github.com/Cola968/neral-helios/actions/runs/35084877377): eager 32,626; compiled 42,877 synthetic tokens/s.
- [Batch 12/16 sweep](https://github.com/Cola968/neral-helios/actions/runs/35085807321): batch12 compiled 44,160; batch16 eager 30,227 versus compiled **44,839** synthetic tokens/s (3.59 GiB reserved, 63 C). Larger batches showed diminishing returns.
- [First CUDA-Graph A/B](https://github.com/Cola968/neral-helios/actions/runs/35086239657) failed because a cached RoPE cosine/sine tensor was overwritten during graph replay. It preserved the control artifact and skipped later tests.
- [Isolated RoPE-prewarm recovery](https://github.com/Cola968/neral-helios/actions/runs/35086701433) succeeded on both matched pairs. Batch8 standard 43,002 versus CUDA graphs 42,841; batch16 standard 44,815 versus CUDA graphs 44,756 synthetic tok/s. CUDA graphs are compatible after prewarming but **not faster** here. Keep standard compilation as experimental control; do not enable `reduce-overhead` in production. The synthetic probe/comparator still have their historical *50k synthetic diagnostic* fields for interpreting old evidence; they do NOT set the new real-training target.

`probe.py` has explicit opt-in, RTX4070/BF16/architecture checks, <=8192 synthetic tokens/microbatch, large batches restricted to upper-four trainable blocks, GPU free >=9000 MiB and initial <=78 C, runtime <=82 C and <=9 GiB reserved, finite-loss/gradient guards and distinct timed versus setup-inclusive rates. It never promotes or changes checkpoints. `compare_probes.py` checks matched synthetic A/B JSON arithmetic and safety but does not establish numerical parity or training qualification. See `GPU_SHAPE_RESULTS_2026-09-16.md` and `GPU_CUDAGRAPH_RESULTS_2026-09-16.md` for historical results.

## Quality and next checkpoint

The existing `sft_contract.py` and masked-EOS local own-checkpoint `train_sft_pilot.py` are written but the SFT pilot has NOT run. Before using external instruction data, independently verify source licensing, privacy/PII, contamination and near-duplicates; do not use foreign pretrained model weights. Baseline and candidate need fixed-seed free generation, EOS, math, code, German/English, science, held-out validation and **complete NIS2** (the earlier 140-case diagnostic is not NIS2 in full). No new quality gains are demonstrated.

The actual long-run exactly resumable NextGen trainer and measured real-data end-to-end profiler do **not** yet exist, so 44k in real training is unverified; historical V8.1 live data nearer 3.5k tokens/s cannot be equated with synthetic compute-only measurements. The objective change alone does not speed up training.

The [CPU evidence CI](https://github.com/Cola968/neral-helios/actions/runs/35087185713) previously passed 48 synthetic contract tests before the 44k change; consult a NEW run for revised tests. No paid HF jobs, dataset/weight uploads or Composio access were used. HF `OpenAssistant/oasst2` metadata was research-only. PR stays draft and unmerged.
