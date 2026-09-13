# Isolated speed-equivalence experiment V5

Status (2026-09-13): implementation and CPU tests completed; RTX 4070 benchmarks,
BF16/compiled equivalence and real-data quality canaries **not run**. The protected
Scale V4 run 34755193140 was still in progress when inspected. Nothing has been
adopted into production. 47–50k tokens/s is a target, not a measured result.

## Isolation and fixed baseline

- Branch: `experiment/speed-equivalence-v5`, based on the active run's exact commit
  `7da6a7e906a17f0acb66c6c0c6d120207f335a63`.
- Existing model, trainer, Scale V4 controller, request, gates and workflows are
  unchanged. `speed_v5_production_blobs.json` pins ten relevant production files;
  the harness rejects drift before starting.
- `train_speed_v5.py` is an explicitly separate experiment copy of that trainer.
  Original `train_guarded_v9.py` is used for timed A trials and long A canaries.
- No SAFE/SHADOW discovery, promotion, restoration, state writes, telemetry issue
  posts, runner restarts, WSL shutdown, or shared-cache modification.
- A live authenticated GitHub check refuses GPU work until protected run
  34755193140 is completed. An additional process/memory/temperature check refuses
  a busy GPU. Direct invocation of the experiment trainer has the same guard.
- Output must be fresh and outside protected assets. Input anchor SHA is explicit;
  never select a changing `resume.pt`. Hashes of the anchor and state files are
  checked again at completion. A mismatch stops the experiment, without restoring
  files that may belong to another run.

## Implemented candidates, in priority order

1. QKV and SwiGLU gate/up projections combined with replay fusion (Pack9).
2. The same combination with one pinned host-to-device transfer per update and
   one consolidated post-step scalar read. The pre-optimizer finite check and
   post-optimizer parameter checks remain mandatory on every update.
3. Mixed-domain physical Pack10, Pack11 and Pack12 within one optimizer update.

The historical `model350_fast.py` also changes RMSNorm and parameter layout.
V5 isolates projection fusion using the **original** RMSNorm, Parameter objects,
canonical state_dict keys, FP32 masters and per-parameter AdamW state. Differentiable
concatenations feed the fused projections. Their overhead might erase the expected
gain; only the compiled GPU benchmark can decide. V5 does not claim to reproduce
the historical fused-layout speed result.

Logical geometry stays exactly 4 rows × 512 tokens × 12 microbatches = 24,576 tokens
per full update. Three replay microbatches retain all three domains and their exact
boosted loss weights. All data identities, sampling order and batch-chain hashes
stay fixed. Main data is sampled in the original Pack9 shape before execution is
regrouped, including partial-tail masking.

Setting production `--main-pack` above 9 alone would be a no-op. V5's separate
`--execution-pack 10|11|12` joins main and replay rows into larger physical forwards,
using per-token weighted cross entropy. This changes reduction order, not the
objective or update boundaries, and must pass the same numerical gates. It may
OOM; failing candidates are not promoted. No larger logical batch, reduced replay,
different data, or changed trainable layers is allowed. Projected-rescue mode is
deliberately rejected by the experiment trainer; its production implementation is
unchanged because replay-reference gradients cannot be fused away in that mode.

## Measurement and acceptance protocol

The default recipe comes directly from Scale V4 `scaled_recipe`: balanced, a 5M
stage, pressure 0. The CLI also accepts the existing retention recipe and bounded
stage size/pressure, using the same LR safety function. Each A/B pair has identical
anchor, dataset SHA checks, seed, token budget, token-progress cosine schedule,
replay boost, last four trainable blocks, clipping and optimizer settings.

1. Run three-update **diagnostic** A/B trajectories for every candidate. Compare
   full trainable gradients, weighted loss, sampled logits keyed by input row,
   final FP32 masters and exact sampling chains. A separate original-production
   control checks the instrumented A trajectory. Diagnostic CPU copies and gradient
   files are never reported as throughput measurements.
2. Reject unless every audited step has loss absolute delta ≤0.002 and gradient
   relative L2 ≤0.005 (the prior V2/V3 thresholds). Additional V5 bounds: sampled
   logit relative L2 ≤0.005 and max absolute delta ≤0.125; final master max absolute
   delta ≤0.00002. NaN/Inf, missing evidence and mismatched identities fail closed.
   Logits are sampled, not exhaustively compared; CPU tests compare full logits on
   a reduced model. These checks do not substitute for full-model quality gates.
3. Three independent timing trials per surviving variant and original production
   A, 64 updates each, excluding eight warmup updates. Reverse trial order on
   alternate repetitions. Record median, p10, sample count, allocated/reserved
   VRAM and end-to-end throughput including compile/startup/checkpoints. Record
   source and data identities. No multiplication of isolated speedup estimates.
4. Require p10 ≥90% of median, allocated VRAM ≤10.25 GiB and reserved ≤11 GiB.
   Select the fastest stable measured improvement. No improvement means no adoption.
5. Run original A and winning B for 5,013,504 tokens each from the same anchor.
   Run unchanged fixed validation (32 batches, seq512, batch4, seed351337), full
   140-case dev and 400-case bilingual evaluations on the anchor and both outputs.
   Invoke existing `core_gate` and `bi_gate`, checking A vs anchor, B vs anchor,
   and B vs A; missing/nonfinite/mixed-checkpoint reports fail closed.
6. Report SAFE strict improvement separately using the existing controller's
   criterion. A tied quality result is never labelled a SAFE promotion. Raw
   candidates are evaluated without interpolation here; this harness does not
   replace the Scale V4 promotion controller or its full selection procedure.
   Production adoption is always false. Checkpoint evaluation compatibility is
   tested; production resume compatibility is not certified by this experiment.

## Current evidence

| Evidence | Result | Scope |
|---|---|---|
| CPU tests, PyTorch 2.8.0+cu128, GPU hidden | 9 passed | Reduced model; 3 AdamW steps; full logits, gradients, frozen weights, masked tails, Packs9–12; harness refusal and evidence checks |
| Production source identity | 10 pinned source files unchanged | Git blob hashes, LF normalized |
| RTX 4070 BF16/compiled numerical audit | Not run | Blocked by protected active training |
| RTX 4070 sustained throughput | Not measured | 47–50k is the requested target |
| Real-data ≥5M-token A/B quality gates | Not run | Required before adoption |
| Speed estimates | None | Historical conversation figures are not new measurements |

Local validation:

```sh
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_speed*v5.py' -v
```

The CPU workflow runs only on GitHub-hosted Ubuntu when this experiment branch
changes. It cannot use the training GPU. The separate GPU workflow is **manual
only** and has not been dispatched. GitHub only exposes `workflow_dispatch` once
the workflow exists on the default branch; do not merge to main just to run it.
Until then, use the guarded CLI below in the existing WSL runtime after training
finishes. No background scheduling has been installed.

```sh
# GITHUB_TOKEN must already be set in the environment (contents/actions read).
# All input paths are explicit, read-only assets; output must be new and separate.
~/.venvs/neral-triton-py312/bin/python frontier350/benchmark_speed_v5.py \
  --assets /mnt/c/path/to/neral-assets \
  --anchor /mnt/c/path/to/immutable-accepted-checkpoint.pt \
  --anchor-sha <verified-64-character-sha256> \
  --out /mnt/d/NeralSpeedV5/unique-experiment-directory
```

Allow at least 60 GiB of free scratch disk. Reports/logs/metrics remain in the
experiment directory; diagnostic gradient files and checkpoints are retained for
inspection. The manual workflow uploads reports/logs/metrics only. Its checkout
and cache paths are unique to the experiment, with no cleanup of other runs.
