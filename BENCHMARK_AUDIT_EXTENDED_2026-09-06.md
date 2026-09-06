# Neral benchmark audit — extended quantitative pass (2026-09-06)

This report derives additional metrics from the already committed raw benchmark reports. It does **not** claim a new checkpoint run: the private checkpoints remain local and a new GPU run requires the self-hosted runner.

## 50M attention optimization across all nine measured training shapes

Source reports: `evidence/reports/throughput50_unpadded.json` and `evidence/reports/throughput50.json`.

| seq | batch | before tok/s | padded-SDPA tok/s | speedup | allocated VRAM reduction | reserved VRAM reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16 | 52,971 | 69,252 | 1.31x | 17.5% | 20.1% |
| 256 | 32 | 53,709 | 76,614 | 1.43x | 20.5% | 18.5% |
| 256 | 64 | 52,295 | 73,343 | 1.40x | 22.7% | 26.0% |
| 512 | 8 | 42,219 | 65,721 | 1.56x | 31.0% | 33.6% |
| 512 | 16 | 42,403 | 74,296 | 1.75x | 35.3% | 35.5% |
| 512 | 32 | 25,951 | 71,305 | 2.75x | 38.6% | 41.7% |
| 1024 | 4 | 22,108 | 65,273 | 2.95x | 48.9% | 50.3% |
| 1024 | 8 | 24,571 | 70,557 | 2.87x | 53.7% | 50.5% |
| 1024 | 16 | 3,358 | 67,589 | 20.13x | 56.9% | 57.2% |

Aggregate diagnostics:

- median speedup: **1.75x**
- geometric-mean speedup: **2.46x**
- median allocated-VRAM reduction: **35.3%**
- median reserved-VRAM reduction: **35.5%**
- the 20.13x result at 1024x16 is an outlier caused by the unoptimized configuration crossing a severe memory/performance cliff; it must not be presented as the typical speedup
- the representative selected training shape, 256x32, improves from 53,709 to **76,614 tokens/s** while reducing allocated VRAM by about **20.5%**

The optimization is numerically justified by the committed CPU FP32 equivalence check: zero-padding head dimension 52 to 56 inside SDPA produced a recorded maximum absolute logit delta of **0.0**, while adding no learned parameters.

## Current 20M vs 50M free-generation regression

Sources: `evidence/reports/free50_baseline20.json`, `evidence/reports/free50_pilot_a.json`, and `evidence/reports/manual_free50.json`.

| Metric | 20M Preview 02 | 50M Pilot A | Interpretation |
|---|---:|---:|---|
| valid UTF-8 | 100% | 100% | tie |
| EOS within 64 byte tokens | 29% | 0% | 50M regression |
| mean repeated trigram fraction | 2.95% | 76.62% | severe 50M collapse |
| exact arithmetic | 0/10 | 0/10 | neither raw model can calculate reliably |
| mean generation speed | 139.23 byte tok/s | 91.24 byte tok/s | 50M is 34.5% slower |
| manual chat score, 0-4 | 1.8 | 1.0 | 50M regression |

The 50M repeated-trigram fraction is approximately **25.93x** the 20M value. That is the strongest current evidence against promoting the 50M pilot.

The generation-speed comparison should not be treated as a final architecture verdict. Current autoregressive generation recomputes the full prefix for every emitted byte and has no KV cache, so decode cost scales poorly with both context and model size.

## Training-stage timing implication

Pilot A processed **999,424** tokens. A cumulative 5,000,000-token Stage B therefore adds **4,000,576** tokens.

Using the actually measured Pilot A optimizer throughput of 75,507 tokens/s gives about **53 seconds** of pure update computation. Scaling the full Pilot A wall-clock ratio (which included its evaluation/checkpoint overhead) gives roughly **74 seconds**. These are extrapolations, not promises: Stage B may differ due to evaluation, disk I/O, thermals, background load and checkpoint behavior.

This is important operationally: on the measured RTX 4070 setup, moving from ~1M to 5M tokens is cheap enough that the correct strategy is **frequent quality gates**, not one blind long run.

## New benchmark coverage prepared on `train-50m-stage-b`

`benchmark_v3.py` adds:

- base-LM continuation separately from chat formatting
- DE/EN grammar continuation preference
- simple facts and commonsense
- raw-model math and code preferences
- true raw-byte BPB with special targets excluded
- context-length BPB at multiple sequence lengths
- CUDA forward throughput and VRAM
- EOS, UTF-8, output uniqueness and repetition diagnostics

`benchmark_v4.py` adds:

- broader German and English grammar
- physics
- astronomy
- aviation
- transport
- mathematics
- logic
- commonsense
- Python/code semantics
- DE/EN translation associations
- induction/pattern completion
- next-token entropy/top-1 probability as a collapse diagnostic
- synthetic in-context recall at approximately 128, 512, 1024, 2048 and 3072 filler bytes
- greedy vs temperature-0.7/top-k-40 base-generation comparison
- BPB through 2048-byte windows
- forward runtime/VRAM

These internal suites intentionally do not call SOLVE, the security guard or other Helios tools for raw-model scores.

## Next acceptance gate

After the self-hosted runner is connected, first run `mode=benchmark` against the existing local 50M Pilot A checkpoint. That establishes v3/v4 baseline numbers **before** Stage B. Then run `mode=train_and_benchmark` to cumulative 5M tokens and compare the same frozen metrics.

Do not promote 50M unless at minimum:

1. repetition collapses substantially from the current 76.62%,
2. EOS behavior returns,
3. base-LM continuations become visibly coherent,
4. pair-preference accuracy rises above chance across several domains rather than one narrow category,
5. true byte BPB improves on held-out data without a context-length pathology,
6. raw outputs, not only loss, improve.
