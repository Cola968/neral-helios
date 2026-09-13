#!/usr/bin/env bash
set -euo pipefail
cd "$NERAL_V5_REPO"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
exec "$HOME/.venvs/neral-triton-py312/bin/python" -u frontier350/benchmark_speed_v5.py \
  --assets "$NERAL_V5_ASSETS" --anchor "$NERAL_V5_ANCHOR" \
  --anchor-sha "$NERAL_V5_ANCHOR_SHA" --out "$NERAL_V5_OUT"
