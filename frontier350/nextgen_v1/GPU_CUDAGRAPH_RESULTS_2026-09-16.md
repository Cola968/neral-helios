# CUDA Graph A/B — resolved compatibility, no speed win (2026-09-16)

All entries here are **fixed-synthetic-batch compute-only** rates on the self-hosted RTX4070, not accepted training, model-quality or release evidence. Both model source SHA256 `d88b187d33cc0c219074c0512f5c89aad023d0694f5f87232c1bf658fad0045a` and original architecture (349,825,152 total params; 55,158,272 trainable upper four blocks) remained unchanged. No real Neral model checkpoint was loaded or modified.

## Failure reproduced and isolated fix

[Initial comparison run 35086239657](https://github.com/Cola968/neral-helios/actions/runs/35086239657): matched batch8 `compile-default` passed at 42,906.527 synthetic tokens/s, but `compile-overhead` failed during warmup with PyTorch `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`. Traceback identifies `model350.py::_rope_table` cos/sin outputs. Batch16 correctly skipped and one baseline JSON was uploaded. This is a CUDA graph cache-lifetime incompatibility, not evidence that ordinary generation is broken.

Commit `6c475996910863f6db4ac593f0c9539eb40cc0ba` changed only the **isolated research `probe.py`**: for `compile-overhead`, precompute the identical cached BF16 RoPE cosine/sine table on the actual model CUDA device outside graph capture, assert its cache identity and gradient-free state, then compile. No change to `model350.py`, tokenizer, weights or production training. This fixes the specific failure for tested configurations; numerical parity/production CUDA graph correctness still require separate tests.

## Complete read-only synthetic result

[Recovery run 35086701433](https://github.com/Cola968/neral-helios/actions/runs/35086701433), [six JSON artifact files](https://github.com/Cola968/neral-helios/actions/runs/35086701433/artifacts/10442926339): safe GPU idle preflight, Python syntax and 11 isolated shape/comparator tests, both default/overhead A/B pairs, six reports and artifact upload all succeeded. Fixed synthetic repeated batch, seq512, 8 warmups, 32 timed optimizer updates per candidate; timing excludes setup and compile, but their elapsed is reported separately. Identical model SHA, trainable count and timed token budget within each pair.

| Batch | Mode | Timed synthetic tok/s | Setup + warmup seconds | Setup-inclusive synthetic tok/s | Peak reserved GiB | Peak temp C |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | compile-default | 43,001.829 | 63.222 | 1,977.857 | 2.385 | 59 |
| 8 | compile-overhead, RoPE prewarmed | 42,840.708 | 35.012 | 3,442.821 | 2.484 | 63 |
| 16 | compile-default | **44,814.674** | 62.983 | 3,808.421 | 3.590 | 64 |
| 16 | compile-overhead, RoPE prewarmed | 44,755.890 | 36.272 | 6,222.321 | 3.506 | 64 |

Strict comparator outputs `ab_b8.json` and `ab_b16.json`: `candidate_faster=false` at both shapes; candidate rate relative to default 0.9963x (batch8) and 0.9987x (batch16). **Neither target reached 50,000 synthetic tokens/s.** The setup-inclusive metric includes one-time compile/setup but not corpus/checkpoint/validation; it is for this deliberately very short probe only. Setup times may be affected by previously compiled/cache-state differences and should not be credited as a reproducible CUDA graph speedup without cold-cache repeat and a long matched realistic workload. A small synthetic-loss difference is not numerical parity or evidence of quality. Max observed temperatures and VRAM remained below unchanged guards.

## Resulting engineering decision

Keep standard `torch.compile(default)` as the baseline for this isolated microbenchmark. Treat prewarming only as a demonstrated compatibility fix; **do not enable reduce-overhead in production** or claim 50k. Further batch inflation returned diminishing gains (see `GPU_SHAPE_RESULTS_2026-09-16.md`). Next performance work needs actual profiling of data loading, synchronization, optimizer, checkpointing, validation and accepted-token accounting on a trusted *own* checkpoint, plus numerical parity and exact resume. No full replacement trainer, real dataset SFT, independent quality gain, model promotion or new 2.5B training started.

Safety/provenance: no checkpoints written in any synthetic probe, no HF jobs/payments/uploads; original V8.1 ended with failure without a new resume. Hugging Face `OpenAssistant/oasst2` was checked only for dataset metadata; separate source rights/privacy/contamination screening is mandatory before ingestion. Composio connection returned `FORBIDDEN` in this conversation; no bypass attempted.
