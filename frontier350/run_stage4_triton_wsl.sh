#!/usr/bin/env bash
set -euo pipefail

REPO=${1:?repo path required}
TRAIN=${2:?train path required}
OUT=${3:?output path required}
VENV="$HOME/.venvs/neral-triton-py28"
CACHE="$HOME/.cache/neral_inductor"

mkdir -p "$HOME/.venvs" "$CACHE"

if [ ! -x "$VENV/bin/python" ]; then
  if ! python3 -m venv "$VENV"; then
    echo "python3 -m venv unavailable; bootstrapping virtualenv without sudo"
    python3 - <<'PY'
import urllib.request
urllib.request.urlretrieve('https://bootstrap.pypa.io/virtualenv.pyz', '/tmp/neral-virtualenv.pyz')
PY
    python3 /tmp/neral-virtualenv.pyz "$VENV"
  fi
fi

P="$VENV/bin/python"

if ! "$P" -c 'import numpy, torch, triton; assert torch.__version__.startswith("2.8."); assert torch.cuda.is_available()' >/dev/null 2>&1; then
  "$P" -m pip install --upgrade pip wheel setuptools
  "$P" -m pip install --upgrade numpy
  "$P" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  "$P" -c 'import triton' >/dev/null 2>&1 || "$P" -m pip install triton==3.4.0
fi

export TORCHINDUCTOR_CACHE_DIR="$CACHE"
export PYTHONUNBUFFERED=1

"$P" - <<'PY'
import torch, triton
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('triton', getattr(triton, '__version__', 'unknown'))
print('gpu', torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
PY

cd "$REPO"
"$P" -m py_compile frontier350/model350.py frontier350/model350_fast.py frontier350/speed_triton350.py
"$P" - <<'PY'
import sys
sys.path.insert(0, 'frontier350')
from model350 import Neral350M
from model350_fast import Neral350MFast
a = Neral350M().parameter_report()
b = Neral350MFast().parameter_report()
print(a, b)
assert a['total'] == 349_825_152
assert b['total'] == 349_825_152
PY

mkdir -p "$(dirname "$OUT")"
"$P" -u frontier350/speed_triton350.py \
  --train-data "$TRAIN" \
  --out "$OUT" \
  --seq 512 \
  --steps 8 \
  --warmup 3
