#!/usr/bin/env bash
set -euo pipefail

REPO=${1:?repo path required}
TRAIN=${2:?train path required}
OUT=${3:?output path required}
VENV="$HOME/.venvs/neral-triton-py312"
BOOT="$HOME/.venvs/neral-uv-bootstrap"
CACHE="$HOME/.cache/neral_inductor"

mkdir -p "$HOME/.venvs" "$CACHE"

# Ubuntu 26.04 exposes Python 3.14 by default, while the locked PyTorch
# 2.8.0/cu128 baseline has no cp314 wheel. Bootstrap user-local Python 3.12
# via uv so Stage-4 stays directly comparable without requiring sudo.
if [ ! -x "$BOOT/bin/python" ]; then
  if ! python3 -m venv "$BOOT"; then
    echo "python3 -m venv unavailable; bootstrapping virtualenv without sudo"
    python3 - <<'PY'
import urllib.request
urllib.request.urlretrieve('https://bootstrap.pypa.io/virtualenv.pyz', '/tmp/neral-virtualenv.pyz')
PY
    python3 /tmp/neral-virtualenv.pyz "$BOOT"
  fi
fi
"$BOOT/bin/python" -m pip install --upgrade pip wheel setuptools uv
UV="$BOOT/bin/uv"

if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1; then
  "$UV" python install 3.12
  rm -rf "$VENV"
  "$UV" venv --python 3.12 "$VENV"
fi

P="$VENV/bin/python"

# uv-created environments intentionally may not contain pip. Use `uv pip`
# against the environment explicitly instead of requiring `python -m pip`.
if ! "$P" -c 'import numpy, torch, triton; assert torch.__version__.startswith("2.8."); assert torch.cuda.is_available()' >/dev/null 2>&1; then
  "$UV" pip install --python "$P" --upgrade numpy
  "$UV" pip install --python "$P" torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  "$P" -c 'import triton' >/dev/null 2>&1 || "$UV" pip install --python "$P" triton==3.4.0
fi

# TorchInductor/Triton generates small host-side C/C++ launch stubs. Keep the
# compiler choice explicit so the benchmark cannot silently depend on shell PATH.
export CC="${CC:-$(command -v gcc || true)}"
export CXX="${CXX:-$(command -v g++ || true)}"
if [ -z "$CC" ] || [ -z "$CXX" ]; then
  echo "Stage-4 requires gcc/g++ for TorchInductor" >&2
  exit 43
fi
export TORCHINDUCTOR_CACHE_DIR="$CACHE"
export PYTHONUNBUFFERED=1

echo "cc=$CC"
echo "cxx=$CXX"
"$CC" --version | head -n1
"$CXX" --version | head -n1

"$P" - <<'PY'
import sys, torch, triton
print('python', sys.version)
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('triton', getattr(triton, '__version__', 'unknown'))
print('gpu', torch.cuda.get_device_name(0))
assert sys.version_info[:2] == (3, 12)
assert torch.__version__.startswith('2.8.')
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
