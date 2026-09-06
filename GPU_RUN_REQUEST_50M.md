# Neral-50M GPU run request

Requested on 2026-09-06.

This branch exists only to trigger `.github/workflows/finish-neral50m.yml` on the repository-specific self-hosted Windows RTX runner. The workflow benchmarks Pilot A first, completes the deterministic full-corpus Stage B sweep, benchmarks best-validation and final checkpoints, compares 20M Preview 02 when locally available, and uploads reports only. Private checkpoints and corpus binaries remain local.
