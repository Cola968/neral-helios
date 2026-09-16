# Genuine-data V9 GPU phase profiler — 2026-09-16

**Evidence:** [successful local RTX 4070 run 35120842844](https://github.com/Cola968/neral-helios/actions/runs/35120842844), job 104877921644. Scope is 349,825,152 total Neral-own parameters, of which only upper four blocks / 55,158,272 are trainable. Three optimizer updates of 24,576 tokens each used verified REAL main and replay corpus shards (two main, three replay), the SHA-pinned 1,010,003,584-token SAFE anchor, BF16 and torch.compile(default). **This is NOT native whole-model training or a 2.073B continuation.**

Preflight independently verified SAFE SHA256 `ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c`, locked tokenizer SHA256 `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`, exact corpus shard sets, free GPU memory 10,939 MiB, 48 C, and three idle utilization readings [0,0,0]. The dedicated workflow left the protected checkpoint and data untouched; no model checkpoint was written, promoted or uploaded.

## Measured phase breakdown

| Update | Wall with diagnostic synchronization | Derived diagnostic tokens/s | Forward | Backward | Sample and host-to-device | Optimizer/finite | Peak reserved GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 — compilation/warmup | 85.75879 s | 286.6 | 75.67309 s | 9.87009 s | 0.01747 s | 0.15118 s | 5,792 |
| 2 — PyTorch CPU/CUDA profiler active | 0.61526 s | 39,944.0 | 0.38240 s | 0.20747 s | 0.01321 s | 0.00743 s | 6,378 |
| 3 — profiler off, stage synchronization remains | **0.56006 s** | **43,881.3** | **0.38237 s** | **0.15602 s** | **0.01270 s** | **0.00690 s** | **6,378** |

Third-step wall share: forward ~68.3%, backward ~27.9%, sample/H2D ~2.3%, optimizer/checks ~1.2%. Small differences from 100% are rounding and other instrumentation. Timing includes explicit GPU synchronization around EVERY phase, but excludes first-step compilation in subsequent-step rates and excludes any checkpoint save, hash, validation, restart or full quality audit. Three steps are too few to establish sustainable throughput, thermals, loss trends or correctness for a multi-hour run. Total actual-data tokens computed in-memory: 73,728. **Durable accepted new tokens: ZERO.**

The profiled step lists `aten::mm` and BF16 GEMM/CUTLASS kernels as leading device operations; `aten::_flash_attention_forward` also appears. Nested profiler events and operator summaries must not be summed into disjoint wall-time fractions. The explicit wall-phase measurements are the relevant attribution. At this exact workload size the candidate bottleneck is GPU-side forward matrix multiplication rather than sample/H2D or the optimizer. No measured GPU Tensor Core occupancy, roofline, memory bandwidth or achievable 60k ceiling is available; Nsight Compute could resolve kernel utilization in a separate read-only, bounded diagnostic.

## Meaning for the 60,000 tokens/s request

60,000 / 43,881.3 = ~1.37 times the measured step rate. This asks for ~36.7% faster step throughput, **not** merely higher idle GPU utilization. Earlier fixed-synthetic compute-only batch16 best was 44,838.848 tokens/s; reducing CUDA graph overhead did not beat compile-default on tested shapes. Improving CPU loading or faster SSD alone targets a roughly 2.3% component here. PyTorch docs specify `torch.compile(reduce-overhead)` may increase memory and is not universally applicable; NVIDIA notes CUDA graphs chiefly help CPU-launch-bound workloads. Previous microbenchmark already found no steady-step speed win. Avoid claiming that max-autotune, alternative BF16/TF32 policy, FlashAttention or bigger batches can reach 60k without controlled same-data and gradient/resume-parity A/B evidence.

## Next falsifiable work (research only)

1. With the original math policy frozen, use PyTorch Profiler `record_shapes` in a short separate run or Nsight Compute to identify top GEMM input dimensions and whether arithmetic throughput, occupancy or memory traffic dominates. Do not publish raw dataset/model traces.
2. Benchmark ONE isolated GEMM/compile change at a time with matched 24,576-token batches, data chain, seed and exact-restart/numeric acceptance. Collect cold-start compile time, all update times, checkpoint serialize+SHA, validation and full wall. Keep unchanged code path as control; stop if quality or state parity regresses.
3. For a *real full-model* run independently solve optimizer warm-start compatibility, VRAM headroom, multi-GB checkpoint crash recovery and bounded journal retention. The current 4-block V9 optimizer is not the full-model optimizer. Release requires separate held-out quality, generation/EOS and NIS2 gates.

**No permanent training, production promotion, 60k claim, full-model proof or public release authorization follows from this diagnostic.**

## Source documentation (engineering references, not performance evidence for Neral)

- PyTorch Profiler: https://docs.pytorch.org/docs/stable/profiler.html
- PyTorch performance tuning / Tensor Cores: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- NVIDIA Nsight Compute Profiling Guide (roofline and workload analysis): https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- NVIDIA CUDA graph performance troubleshooting: https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html
