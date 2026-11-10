# Neral performance + data sprint — 2026-09-08

## Ambition

Target at least **3.0x effective development throughput** versus the previous manual/serial workflow.
This means more useful, benchmarked model progress per wall-clock day, not a promise that every
single CUDA kernel or training run becomes 3x faster.

The 3x target is decomposed into independent levers so each claim can be measured:

1. **GPU training throughput** — architecture/kernel/data-loader changes;
2. **GPU occupancy time** — keep prepared work queued and resume-safe;
3. **data throughput** — sharded preparation and preprocessing off the critical GPU path;
4. **experiment efficiency** — kill weak variants early instead of spending full token budgets;
5. **evaluation latency** — automatic frozen reports immediately after meaningful checkpoints.

A 200% improvement means a 3.0x final ratio (old throughput + 200%). The project must report the
actual measured ratio and must not relabel a smaller gain as 200%.

## Changes in this branch

### Existing Neral-50M

- Preserve current I1 architecture/checkpoint compatibility.
- Add an exact-weight fused SwiGLU runtime path:
  - current 50M already has fused QKV;
  - concatenate `gate` + `up` projection weights;
  - one gate/up GEMM instead of two;
  - linear GEMM calls per 8-layer forward: **40 -> 32 (-20%)**.
- The fused layout is not injected into an interrupted optimizer resume because AdamW parameter
  state layout differs. It is safe after legacy checkpoint loading for inference/evaluation and
  for future phases that intentionally initialize a fresh optimizer.

### Neral Orion ~100M

Orion is the model-generation name; Helix remains the research lineage.

New throughput-first candidates:

- **Orion A3**: 100,002,304 parameters; 14 layers; width 768; 12 Q heads / 2 KV heads; head dim 64;
  SwiGLU 2496; context ceiling 8192.
- **Orion A4**: 100,103,936 parameters; 16 layers; width 768; 12 Q heads / 2 KV heads; head dim 64;
  SwiGLU 2112; context ceiling 8192.

Both use:

- fused QKV projection;
- compact GQA;
- fused gate/up SwiGLU projection;
- Q/K RMSNorm;
- cached RoPE;
- tied byte embeddings;
- optional activation checkpointing.

Relative to legacy Helix A2's 24 blocks and 7 separate linear GEMMs/block, A3 has 14 blocks and
4 linear GEMMs/block: **168 -> 56 linear GEMM calls/forward (-66.7%)**. This is a structural
kernel-launch reduction, not yet a measured 3x training speed result.

## Mandatory RTX-4070 measurement

`probe_perf_v2.py` measures:

- BF16 forward;
- cross entropy;
- backward;
- gradient clipping;
- fused AdamW update;
- real CUDA-synchronized step time;
- tokens/s and tokens/hour;
- peak allocated/reserved VRAM;
- OOMs instead of silently lowering the workload.

Matrix:

- 50M legacy;
- 50M fused-MLP;
- Orion A3;
- Orion A4;
- sequence lengths 256 / 512 / 1024 / 2048 / 4096;
- approximately 8192 byte tokens/update;
- activation-checkpoint variants at long context for Orion.

No architecture is promoted from theoretical FLOPs/parameter count alone.

## Data-scale program

See `DATA_SCALE_V2.md`.

The immediate path is staged rather than blindly feeding 10B tokens into the current 50M:

- finish 50M I1 at 500M cumulative language tokens plus guarded RawMath/SFT;
- H1 to 1B only if frozen evidence remains positive;
- optional 50M H2 to 2B while held-out/NIS2 slopes justify it;
- Orion 100M starts with a multi-billion-token curriculum and can extend toward 5B;
- 500M-generation long-term target: >=10B Neral byte tokens.

Preferred public candidate pool audited through Hugging Face discovery:

- FineWeb2 DE/EN;
- FineWeb-Edu EN;
- FineMath after existing fail-closed audit;
- OASST2 for instruction tuning;
- code data only after source-level licensing/provenance admission.

## Promotion criteria

### Performance

- report real RTX 4070 tok/s for every candidate;
- prefer Pareto-efficient speed/VRAM configurations;
- target >=1.5x architecture throughput improvement in the next decision round;
- 3.0x remains the stretch goal for the combined engineering workflow, not a fabricated minimum.

### Quality

Speed cannot compensate for capability loss. Continue to require:

- frozen validation CE/BPB;
- NIS2 and components;
- math transfer/OOD;
- code/reasoning;
- context/needle recall;
- EOS/repetition/UTF-8 health;
- forgetting checks.

### Data

- immutable revisions;
- license/provenance manifest;
- exact + near dedup;
- benchmark contamination exclusion;
- document-bounded packing;
- measured accepted/rejected counts;
- raw corpora/checkpoints remain local.

## Current limit

The connected Hugging Face account is authenticated but not Pro, so Hugging Face Jobs is not
used as paid GPU compute here. Hugging Face is being used for dataset/research discovery and the
self-hosted RTX 4070 remains the training/profiling hardware.
