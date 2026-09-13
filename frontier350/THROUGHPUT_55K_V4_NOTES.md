# 55k throughput optimization notes

Current real Scale V4 sustained median: about 42.5k tok/s on RTX 4070.

Do not interrupt the active quality run. After it finishes, benchmark these candidates in order:

1. Productionize the already-passing fused QKV/SwiGLU execution candidate.
2. Productionize replay fusion only if the previous numerical-equivalence tolerance remains satisfied.
3. Sweep equivalent update shapes at constant 24,576 tokens/update to improve Tensor Core occupancy.
4. Probe disabling activation checkpointing only if peak allocated memory remains below 11 GiB and no OOM occurs.
5. Compare compile modes after warmup and measure sustained median/p10, not short peaks.

Acceptance:
- >=50k sustained is a major win.
- >=55k sustained is the target.
- p10 >=90% of median.
- no NaN/Inf/OOM.
- frozen weights unchanged.
- token accounting exact.
- quality gates unchanged.
- real >=5M-token canary before Scale V4 adoption.

Never combine isolated percentage gains and call the product proven; only the combined measured run counts.
