# Neral 350M — Throughput 55K V4 plan

Goal: raise sustained training throughput on the RTX 4070 from the current ~42.5k tok/s toward 55k tok/s without weakening training quality or changing token accounting.

## Principles

- 55k is a target, not a claim. Only sustained measured throughput counts.
- Keep the currently running Scale V4 job untouched.
- Benchmark changes on a dedicated branch before production adoption.
- Preserve exact optimizer/token semantics unless a candidate is explicitly verified equivalent.
- Reject any candidate with OOM, frozen-weight drift, unstable p10 throughput, or numerical mismatch outside the existing tolerance contract.

## Candidate order

1. Integrate the already-passing fused QKV/SwiGLU execution path from the Throughput-60K V3 probe.
2. Integrate replay-fusion where numerical equivalence remains within the existing probe tolerance.
3. Sweep batch/microbatch shapes at constant tokens-per-update (24,576) to use RTX 4070 tensor cores more efficiently.
4. Probe disabling activation checkpointing where VRAM remains < 11.0 GiB; this is a benchmark-only candidate until memory and numerical behavior are proven.
5. Compare torch.compile default vs max-autotune only after warmup/compile cost is amortized.
6. Adopt only the fastest stable combination, then run a real >=5M-token canary before Scale V4 production use.

## Success criteria

- sustained median >= 50k tok/s is a major improvement;
- sustained median >= 55k tok/s is the stretch target;
- p10 >= 90% of median;
- peak allocated VRAM < 11.0 GiB;
- no OOM / NaN / Inf;
- frozen layers bit-identical;
- token accounting exact;
- validation/dev/bilingual gates unchanged.

## Production rule

No throughput optimization is allowed to replace the current Scale V4 execution path while a quality run is active. A candidate is production-eligible only after a standalone benchmark and a real training canary both pass.
