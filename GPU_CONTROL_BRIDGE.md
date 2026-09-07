# Neral GPU Control Bridge

This bridge lets the private self-hosted Windows runner execute a deliberately small set of Neral diagnostics from GitHub while keeping checkpoints and corpora local.

## Security boundary

Version 1 is **read-only with respect to model weights**. It does not accept arbitrary shell commands, file paths, training targets, Python snippets, or model-changing actions. The only accepted actions are:

- `gpu_status` — CUDA/BF16/GPU/VRAM/temperature/power/utilization report.
- `inspect_b2` — read the B2 summary and hash local B2 checkpoint files without loading or modifying weights.
- `benchmark_b2` — run the existing fail-closed completed-B2 100M frozen benchmark handoff.
- `run_tests` — execute the repository pytest suite.

Private assets stay under `NERAL_ASSETS_ROOT`. The workflow rejects `.pt`, `.pth`, `.bin`, `.safetensors`, `.parquet`, and `.gz` files from report uploads. GitHub receives only small logs/reports.

## Runner requirements

The repository runner must be online with labels:

```text
self-hosted
windows
x64
neral-gpu
```

The Windows account running `run.cmd` must expose:

```text
NERAL_ASSETS_ROOT
NERAL_PYTHON
```

See `GPU_RUNNER_SETUP.md` for initial runner registration.

## Trigger paths

There are two supported trigger paths:

1. Manual: **Actions -> Neral GPU Control Bridge -> Run workflow**.
2. Controlled request branch: pushing only `gpu_requests/request.json` on branch `gpu-control-run` triggers the bridge. The runner checks out executable control code from `main`; the trigger branch supplies only the JSON request.

Request schema:

```json
{
  "version": 1,
  "request_id": "b2-status-001",
  "action": "inspect_b2"
}
```

`request_id` is restricted to 1-64 ASCII letters, digits, `.`, `_`, or `-`. Unknown fields and unknown actions fail closed.

## Concurrency

The bridge shares the `neral-50m-b2-100m` GitHub Actions concurrency key with B2/campaign workflows so GitHub-managed benchmark work cannot overlap a GitHub-managed B2 writer. A separately launched local/Astra training process is outside GitHub's concurrency system; do not request `benchmark_b2` until that local B2 writer has finished and `summary.json` reports `complete=true` and `stop=target_reached`.

## Audit trail

Every request writes a normalized `request.json` plus `result.json` and action-specific logs/reports. Reports are persisted locally under `NERAL_ASSETS_ROOT/reports/gpu_control/<github-run-id>` and uploaded to GitHub Actions for short-term inspection.
