# Neral 350M Scale V3 — Success Controller

Scale V3 is the continuation framework for moving Neral 350M from the current safe ~1.01B-token region toward 10B tokens. Its two explicit objectives are:

1. raise the probability that each training round yields a **safely promotable** checkpoint;
2. spend far less wall time evaluating candidates that were unlikely to pass anyway.

The acceptance standard itself is **not relaxed**. Every promoted checkpoint still has to preserve the complete 140-case recovery-dev floor, fixed validation loss, the 400-case bilingual overall score and both bilingual directions, while also producing at least one strict improvement.

## Why V2 was inefficient

Scale V2 trained multiple learning rates and then evaluated a large alpha family. The actual 1–2M-token training blocks were short at ~40k tok/s, while repeated full dev/bilingual evaluations dominated wall time. More importantly, changing only LR after a retention failure did not directly attack the failure mechanism.

V3 therefore moves most of the adaptation **before and inside training**, and uses cheap evaluation only for ranking.

## 1. Start with a calibratable 1M rung

The ladder is now:

`1M -> 2M -> 5M -> 10M -> 25M -> 50M -> 100M -> 250M -> 500M`

V2 showed that 2M was not yet a reliable first hop from the 1.010B anchor, so V3 begins at 1M. The first successful full 1M promotion immediately unlocks 2M. From 2M upward, two repeatable full-horizon promotions are required before increasing the block size again.

A midpoint promotion is useful progress but does not count as proof that the full block size is stable.

## 2. Retention-first recipes

The first run no longer begins with the most aggressive recipe. Until the model proves repeatable promotions, the default is `retention`:

| Recipe | Start LR | End LR | Replay boost | Mode |
|---|---:|---:|---:|---|
| balanced | 3.0e-5 | 1.2e-5 | 1.45 | replay-only fast path |
| **retention** | **2.2e-5** | **8.0e-6** | **1.65** | replay-only fast path |
| projected_rescue | 1.5e-5 | 5.0e-6 | 1.90 | replay-gradient projection |

The LR decays with a cosine-by-token schedule inside each block. As block size grows, V3 automatically scales the LR downward and replay protection upward. Repeated failures add `failure_pressure`, which makes the next recipe more conservative instead of blindly repeating the same update.

## 3. Free half-horizon salvage

Every train run writes a midpoint snapshot at about 50% of its token budget. V3 evaluates the full endpoint first. If it cannot produce a promotable candidate, the midpoint is available immediately without another compile or another training run.

This means one 2M training pass can provide both a 2M and approximately 1M recovery opportunity.

## 4. Rotating smoke screen before full gates

`eval_recovery_smoke_v3.py` evaluates 35 cases: 5 cases from each of the seven dev categories. There are four deterministic rotations, so consecutive rounds do not keep optimizing against one tiny fixed subset.

The smoke screen is used **only to rank recovery candidates**. It compares:

- correct-to-wrong regressions;
- wrong-to-correct gains;
- option-margin movement;
- an 8-batch validation probe.

It can never promote a model. The top two smoke-ranked recovery candidates at most are sent to the complete immutable promotion gate.

## 5. Learned recovery trust step

Instead of a fixed six-alpha grid, the controller keeps a persistent `preferred_alpha`. For each source checkpoint it screens only:

`preferred_alpha`, `0.5 × preferred_alpha`, `1.6 × preferred_alpha`

bounded to `[0.005, 0.10]`.

After a failed round, the result changes the next trust step:

- regression / bilingual loss -> smaller alpha;
- all floors held but no strict gain -> larger alpha.

After a successful promotion, the winning alpha becomes the next preferred value. This turns repeated runs into a control loop instead of restarting a grid search from scratch every time.

## 6. Hierarchical evaluation

The intended order is:

`training health -> rotating 35-case smoke + 8-batch validation -> full 140-case dev + 32-batch validation -> 400-case bilingual only if core passes -> promotion`

Thus most weak candidates stop before the expensive bilingual holdout. Full promotion evidence remains identical in strictness to the previous controller.

## 7. Failure does not end the campaign immediately

A failed round:

- preserves the existing safe anchor;
- raises retention protection;
- moves toward a safer recipe;
- reduces block size when possible;
- updates the preferred recovery alpha.

The controller stops only after two consecutive failed rounds at the minimum 1M rung. This avoids losing an entire overnight campaign to one unlucky candidate while still failing closed.

## 8. Safe state survives shutdowns

`frontier350_scale_v3_state.json` records only the last fully promoted anchor plus control state:

- safe checkpoint path / SHA / token count;
- block rung;
- success streak;
- conservative recipe level;
- preferred alpha;
- failure pressure;
- full baseline metrics for that exact anchor.

An unverified raw or smoke checkpoint never becomes the safe anchor. After a reboot, the next run resumes from the latest fully promoted checkpoint discovered on disk.

## 9. Throughput geometry

Pack 6 remains the current stable geometry. The trainer supports:

- variable start/end LR;
- cosine token schedule;
- replay boost with normalized loss weights;
- midpoint snapshot;
- fast replay-only path;
- projected rescue mode;
- frozen-weight verification and checkpoint identity contracts.

The target is not merely maximum tok/s; it is **safe promoted tokens per hour**.

## Promotion invariant

Smoke results, training loss, throughput, or a GitHub workflow success status are never enough to promote weights. Promotion requires the complete quality gate for the exact candidate checkpoint. On any failure, the previous promoted anchor stays authoritative.

## Expected effect

This architecture is designed to improve promotion probability through smaller initial horizons, retention-first training, stronger replay, adaptive LR, projected rescue, midpoint salvage and learned trust steps. It is also designed to cut evaluation overhead through smoke ranking and early rejection.

The numerical success rate is deliberately **not claimed in advance**. It must be measured on real RTX 4070 runs. Tomorrow's first V3 run should therefore be treated as calibration evidence: if 1M promotes cleanly, the controller can immediately begin climbing the ladder rather than repeating another LR × alpha matrix.
