# NextGen V1: first RTX 4070 compute probe — 2026-09-16

**Verified source:** GitHub Actions run https://github.com/Cola968/neral-helios/actions/runs/35084101942 , artifact `neral-nextgen-v1-compute-only-35084101942` (JSON files `eager.json`, `compiled.json`). Both GPU probe steps and artifact upload succeeded. The measurements are synthetic, compute-only, and **NOT accepted-token end-to-end training throughput**.

| Setup | Eager PyTorch | `torch.compile(mode='default')` |
| --- | ---: | ---: |
| Synthetic compute-only tokens/s | 24,700.013 | 38,129.010 |
| Timed 12-step wall seconds | 0.99498 | 0.64455 |
| Peak reserved GPU memory (GiB) | 2.23047 | 1.85547 |
| Peak GPU temperature (C) | 52 | 52 |

Same RTX 4070, same Neral architecture SHA `d88b187d33cc0c219074c0512f5c89aad023d0694f5f87232c1bf658fad0045a`, 349,825,152 parameters, **55,158,272 trainable parameters in the upper four blocks only**, batch=4, context length=512, 3 warmups and 12 timed steps. One fixed synthetic batch was reused; runs are separately seeded. Compiled warmup/compilation overhead is **not** included in measured wall seconds. Speedup in this narrow setup approximately 1.544x (54.4%). Neither setting achieved 60,000 compute-only tokens/s at this shape.

**Do not claim training quality from the reported synthetic loss:** There is no real corpus, independent validation, accepted stages, checkpoint/restart timing, NIS2, answer generation, or post-training evidence. Measured throughput is an upper-bound-oriented compute component; the future end-to-end rate can be lower after all overheads and safety gates. No existing weights were loaded or promoted.

**Old run status:** https://github.com/Cola968/neral-helios/actions/runs/35069331704 ended `failure` with `EARLY_STOP_OR_REJECT_LIMIT`; it was not manually canceled or resumed. Its evidence shows `target_reached=false`, 2,073,161,216 final continuity candidate tokens but a distinct last safe checkpoint of 1,010,003,584 tokens. Do not call the 2.073B candidate safe or conflate these two counters. Current V8.2 workflow is a separate read-only quality diagnostic, not a training resume.

**Decision:** Keep production closed. Next experiments: compare larger fixed shapes (batch=8 and seq 512/1024), include matched eager controls and stable warmups, then independently validate complete same-data stages with checkpoint/eval/restart overhead and genuine held-out quality. No promise of 60-70k accepted tokens/s from this probe.