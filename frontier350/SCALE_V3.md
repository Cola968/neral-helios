# Neral 350M Scale V3

Scale V3 is the continuation framework for moving Neral 350M from the current safe ~1.01B-token region toward 10B tokens without spending most wall time on exhaustive LR/alpha grids.

## Primary objective

Increase the probability that a trained block produces a promotable checkpoint **without weakening the existing quality floor**. A promotion still requires no regression in the 140-case development suite, no validation-loss regression, no bilingual overall/direction regression, and at least one strict improvement.

## What changes versus Scale V2

### 1. Training is made safer before evaluation

V2 tried several constant learning rates after a block had already shown retention problems. V3 changes the update itself:

- cosine LR decay within every block;
- replay loss is explicitly boosted and normalized;
- larger blocks automatically receive a smaller LR;
- replay protection rises gradually with block size;
- Pack 6 remains the throughput geometry unless later evidence replaces it.

This targets the actual failure mode (small reasoning/bilingual regressions) instead of merely searching more interpolation coefficients after the fact.

### 2. One train run produces two useful horizons

The trainer writes an in-process midpoint checkpoint at ~50% of the block. If the full block fails, V3 can evaluate the midpoint immediately. There is no second training pass, no second compile and no duplicated GPU work just to obtain a 1M fallback from a 2M run.

### 3. Sequential trust-step search replaces the alpha grid

V2 evaluated a fixed list of interpolation alphas. V3 starts from one trust step and reacts to the failure:

- regression -> shrink alpha;
- all floors held but no measurable improvement -> grow alpha;
- hard cap of three attempts.

The expensive bilingual holdout runs only after validation/dev already pass.

### 4. Adaptive recipes replace the LR matrix

There are three retention recipes: `balanced`, `retention`, and `rescue`. At most two are trained in one round. If a round succeeds, the controller moves back toward the faster recipe. If it fails, the next run becomes more conservative.

### 5. Block size grows from evidence

The ladder is:

`2M -> 5M -> 10M -> 25M -> 50M -> 100M -> 250M -> 500M`

Two full-block successes are required before moving up one rung. A midpoint promotion does not count as a full-block success. A failed round moves one rung down and increases retention protection.

### 6. Safe state survives reboots and cancelled runs

`frontier350_scale_v3_state.json` is written only after a candidate has passed every promotion gate and has been copied into a selected run directory. If Windows is shut down, an unpromoted raw checkpoint cannot replace the safe anchor. The next V3 run discovers the latest promoted Neral 350M anchor automatically.

### 7. Evaluation overhead is bounded

V3 does not run a full LR x alpha matrix. The controller trains at most two recipes per round, gets the half-block fallback for free, and evaluates at most three sequential trust steps per source checkpoint. The bilingual suite is skipped for any candidate that already fails validation/dev.

### 8. Telemetry understands dynamic blocks

Live telemetry reads the active trainer contract to infer the actual number of updates, so ETA remains meaningful when the block grows beyond 2M. It also reports current LR and replay boost.

## Safety invariant

Scale V3 is designed to raise promotion probability by improving the training trajectory, **not by making the acceptance threshold easier**. If no candidate passes, the last safe promoted checkpoint remains authoritative.

## First-run plan

The first V3 run should start conservatively at the latest promoted anchor found on disk, use the 2M rung, and prove that cosine LR + replay boost + adaptive trust-step recovery can pass the unchanged gate. Only after repeated full-block successes should the controller grow the block size.

The design aims for a much higher useful-training/evaluation ratio than Scale V2, but the actual promotion success rate must be measured on the RTX 4070 runs; it is not assumed in advance.
