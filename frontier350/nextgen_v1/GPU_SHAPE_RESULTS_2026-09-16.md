# NextGen RTX4070 benchmark: batch 8–16 — 2026-09-16

**Source A:** https://github.com/Cola968/neral-helios/actions/runs/35084877377 , artifact `neral-nextgen-v1-batch8-synthetic-35084877377`.
**Source B:** https://github.com/Cola968/neral-helios/actions/runs/35085807321 , artifact `neral-nextgen-v1-50k-synthetic-35085807321` (3 JSON records). Both GitHub workflows completed successfully and saved JSON evidence. All rates below are **synthetic compute-only, unaccepted** tokens/s, NOT foundation-train end-to-end, quality, NIS2 or production promotion.

| Batch x context | Execution | Timed tokens/s | Timed steps | Setup + warmup (s) | Setup-inclusive tokens/s | Peak reserved GiB | Peak temp C |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 x 512 | eager | 32,626.258 | 20 | n/a (v1) | n/a | 3.176 | 62 |
| 8 x 512 | compile-default | 42,876.721 | 20 | n/a (v1) | n/a | 2.385 | 63 |
| 12 x 512 | compile-default | 44,160.247 | 24 | 63.152 | 2,217.696 | 2.969 | 60 |
| 16 x 512 | eager | 30,227.376 | 24 | 5.788 | 15,994.276 | 4.787 | 63 |
| 16 x 512 | compile-default | **44,838.848** | 24 | 62.378 | 2,944.885 | 3.590 | 63 |

All runs used the **same architecture file SHA256** `d88b187d33cc0c219074c0512f5c89aad023d0694f5f87232c1bf658fad0045a`, 349,825,152 total / 55,158,272 trainable parameters (upper four blocks), RTX4070 BF16, seq=512, fused AdamW, repeated synthetic batch. Prior batch-4 result: compile-default 38,129.010 synthetic tokens/s. Not matched wall conditions across separate workflows; batch16 eager vs compiled was matched within the same workflow. Batch16 compiled is below the 50,000 compute-only target by 5,161.152 tokens/s. None meets the required *accepted* end-to-end objective.

**Diagnosis:** Increasing batch from 12 to 16 improved compiled throughput only 1.54%; do not extrapolate unlimited batch speedups. Setup and compilation can dominate tiny runs (62s on compiled batch16 vs 4.38s timed); real long-running benefit must be evaluated on full wall time. GPU headroom and temperature were within enforced guards; the first test suggests a performance/launch-overhead investigation rather than blindly increasing batch.

**Next falsifiable hypothesis:** Test `torch.compile(mode='reduce-overhead')` against matched `compile-default` for batch8 and batch16 using the *same script and architecture* and a single isolated guarded runner; accept improved code paths only after numerical/gradient parity and real end-to-end accepted-token evidence. Official docs: https://docs.pytorch.org/docs/stable/generated/torch.compile.html and https://huggingface.co/docs/transformers/torch_compile. CUDA graphs may use more memory and may not apply; refuse unsafe memory/thermal conditions. Do not change or publish checkpoints, run paid Hugging Face jobs, or assume actual weights improve from a synthetic benchmark.
