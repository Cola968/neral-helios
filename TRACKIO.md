# Neral training observability with Hugging Face Trackio

Neral uses a local-first telemetry layer. Training always writes numeric metrics to `telemetry.jsonl`; optional Trackio sync can mirror those metrics to a Hugging Face Space for live inspection.

## Privacy boundary

Trackio receives only numeric training telemetry and a small explicit run configuration. Neral corpus rows, prompts, benchmark examples, checkpoints, model weights, local file paths, and private datasets are not logged by `training_observability.py`.

## Runner environment

Remote sync is opt-in. Configure these environment variables on the self-hosted runner before a future run:

```text
NERAL_TRACKIO_ENABLED=1
NERAL_TRACKIO_PROJECT=neral-helios
NERAL_TRACKIO_SPACE_ID=<your-user>/<your-trackio-space>
HF_TOKEN=<token with permission to write that Space>
```

If `NERAL_TRACKIO_ENABLED` is unset, training still writes local JSONL telemetry. If a Space ID is configured without an HF token, the GPU workflow fails before training rather than silently losing remote telemetry.

## Metrics

Neral-100M logs at least:

- training loss
- validation loss when a separate validation stream is configured
- learning rate
- gradient norm
- cumulative tokens
- tokens/second
- effective sequence length and microbatch after OOM fallback
- peak allocated and reserved CUDA memory

Alerts are emitted for CUDA OOM fallback, fatal OOM, large loss spikes, throughput collapse, resume events, and training failure.

Future Neral-50M H1 runs use `trackio_bridge50.py`, which tails only numeric `metrics.jsonl` files under the local runs directory while leaving the existing trainer untouched.

## Important

The H1 run that was already active when this integration was added cannot be instrumented retroactively. The updated workflow applies to subsequent authorized runs only.
