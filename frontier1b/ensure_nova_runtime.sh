#!/usr/bin/env bash
set -euo pipefail

# Dedicated research runtime. This must not mutate the existing Arctus training venv.
VENV="${NERAL_NOVA_VENV:-$HOME/.venvs/neral-nova1b}"
MAMBA_REV="e9594ce1c732d97440f0332fdc43170a2294dbfa"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

"$PY" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Nova requires Python >= 3.10")
try:
    import torch
except Exception as exc:
    raise SystemExit(f"CUDA-enabled PyTorch must already be available: {exc}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible in the Nova WSL runtime")
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
PY

# Mamba-3 is source-only in the current upstream documentation. Pin the exact revision.
# --no-build-isolation is required upstream so the existing CUDA-enabled PyTorch is reused.
if ! "$PY" - <<PY
import importlib.metadata
try:
    import mamba_ssm.modules.mamba3 as m3
    print("mamba3 import already available", m3.__file__)
except Exception:
    raise SystemExit(1)
PY
then
  "$PIP" install --upgrade pip setuptools wheel
  "$PIP" install --no-build-isolation "git+https://github.com/state-spaces/mamba.git@${MAMBA_REV}"
fi

# MIMO rank-4 requires the upstream MIMO kernel path. Fail closed if it is unavailable.
"$PY" - <<'PY'
import torch
from mamba_ssm.modules.mamba3 import Mamba3
m = Mamba3(
    d_model=128,
    d_state=128,
    expand=2,
    headdim=64,
    is_mimo=True,
    mimo_rank=4,
    chunk_size=16,
    is_outproj_norm=False,
    dtype=torch.bfloat16,
).cuda().to(torch.bfloat16)
x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    y = m(x)
if y.shape != x.shape or not torch.isfinite(y).all():
    raise SystemExit("Mamba-3 MIMO runtime smoke failed")
print("Mamba-3 MIMO runtime OK", tuple(y.shape))
PY
