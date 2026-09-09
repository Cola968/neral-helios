# ORION CPU-only architecture hardening — 2026-09-09

## Scope

This sprint was executed while the local RTX 4070 was unavailable. No CUDA training, CUDA benchmark, large local inference or random-weight quality benchmark is authorized by this document.

## Current architecture set

| Candidate | Params | Layers | d_model | Q/KV heads | head dim | FFN | Context | Linear GEMMs / forward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Neral 50M legacy | 50,012,976 | 8 | 624 | 12/12 | 52 | 2496 | 4096 | 40 |
| Neral 50M fused MLP | 50,012,976 | 8 | 624 | 12/12 | 52 | 2496 | 4096 | 32 |
| Helix A1 | 99,783,168 | 32 | 512 | 16/4 | 32 | 1600 | 8192 | 224 |
| Helix A2 | 100,470,912 | 24 | 640 | 10/2 | 64 | 1664 | 8192 | 168 |
| ORION A3 | 100,002,304 | 14 | 768 | 12/2 | 64 | 2496 | 8192 | 56 |
| ORION A4 | 100,103,936 | 16 | 768 | 12/2 | 64 | 2112 | 8192 | 64 |

The GEMM counts are structural call counts for the major Linear projections, not measured kernel launches and not speedup claims.

## Exact parameter partition

Calculated by `orion_static_metrics.py` and guarded by CPU tests:

- Neral 50M: 6,231,264 parameters/block; 1,557,504 attention; 4,672,512 FFN.
- Helix A1: 3,114,048 parameters/block; 655,424 attention; 2,457,600 FFN.
- Helix A2: 4,179,328 parameters/block; 983,168 attention; 3,194,880 FFN.
- ORION A3: 7,128,704 parameters/block; 1,376,384 attention; 5,750,784 FFN.
- ORION A4: 6,243,968 parameters/block; 1,376,384 attention; 4,866,048 FFN.

Fused QKV and fused gate/up change execution structure but not represented parameter count.

## BF16 static weight memory

Calculated only from parameters × 2 bytes:

- 50M: 95.39 MiB
- Helix A1: 190.32 MiB
- Helix A2: 191.63 MiB
- ORION A3: 190.74 MiB
- ORION A4: 190.93 MiB

This excludes gradients, optimizer state, activations, allocator fragmentation, kernels and framework overhead.

## Compact KV cache, batch 1, BF16

| Candidate | 1K | 2K | 4K | 8K |
|---|---:|---:|---:|---:|
| Neral 50M | 19.5 MiB | 39 MiB | 78 MiB | n/a |
| Helix A1 | 16 MiB | 32 MiB | 64 MiB | 128 MiB |
| Helix A2 | 12 MiB | 24 MiB | 48 MiB | 96 MiB |
| ORION A3 | 7 MiB | 14 MiB | 28 MiB | 56 MiB |
| ORION A4 | 8 MiB | 16 MiB | 32 MiB | 64 MiB |

These values are exact tensor-size calculations for the compact stored K/V representation. Temporary kernel workspaces are not included.

## Why A3 remains the throughput-first candidate

A3 is currently the best static throughput hypothesis because it combines:

- 14 layers rather than A2's 24;
- fused QKV;
- fused SwiGLU gate/up;
- 64-wide heads;
- 2 KV heads;
- 56 major Linear GEMM calls/forward versus A2's structural 168;
- only 28 MiB BF16 KV cache at 4K and 56 MiB at 8K.

This does **not** promote A3. A4 remains the depth/balance comparator. Real BF16 forward/backward/optimizer throughput, VRAM and then trained quality are still required.

## Crash/resume hardening added

`checkpoint_io_v3.py` defines a future shared resume envelope with:

- model state;
- optimizer state;
- scheduler state;
- CPU RNG state;
- CUDA RNG state when CUDA exists;
- Python and optional NumPy RNG state;
- step and cumulative token count;
- curriculum phase;
- structured dataset cursor;
- config hash;
- architecture hash;
- dataset-manifest hash;
- source commit;
- same-directory temp file + fsync + atomic rename;
- SHA-256 sidecar;
- `last_good_checkpoint.json` advanced only after final checksum verification;
- path traversal rejection for `last_good` resolution;
- fail-closed refusal to deserialize torch pickle unless the caller explicitly marks it trusted-local.

The module is a contract for integration into future 50M/ORION trainers; existing interrupted campaigns are not silently migrated to a different checkpoint layout.

## Next RTX benchmark contract

The already prepared `probe_perf_v2.py` remains the hardware evidence lane. The next GPU window should run one reproducible matrix over:

- Neral 50M legacy;
- Neral 50M fused MLP;
- ORION A3;
- ORION A4;
- seq 256/512/1024/2048/4096 and 8192 only when feasible;
- BF16 forward, backward, gradient clipping and optimizer;
- checkpointing on/off where methodologically valid;
- synchronized step time, tokens/s, tokens/hour, peak allocated/reserved VRAM and OOM.

Any GPU-utilization metric must identify its sampling source and cadence; it is supplemental because short kernels and asynchronous execution can make utilization sampling misleading.

## Promotion boundary

No ORION variant receives a capability/NIS value until it has trained weights. Random initialization is never a quality benchmark. Static efficiency can only decide which candidates deserve the scarce RTX measurement budget.
