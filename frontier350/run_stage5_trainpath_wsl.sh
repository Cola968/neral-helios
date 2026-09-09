#!/usr/bin/env bash
set -euo pipefail

REPO=${1:?repo path required}
TRAIN=${2:?train path required}
OUT=${3:?output path required}
VENV="$HOME/.venvs/neral-triton-py312"
BOOT="$HOME/.venvs/neral-uv-bootstrap"
CACHE="$HOME/.cache/neral_inductor"

mkdir -p "$HOME/.venvs" "$CACHE"

if [ ! -x "$BOOT/bin/python" ]; then
  if ! python3 -m venv "$BOOT"; then
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
if ! "$P" -c 'import numpy, torch, triton; assert torch.__version__.startswith("2.8."); assert torch.cuda.is_available()' >/dev/null 2>&1; then
  "$UV" pip install --python "$P" --upgrade numpy
  "$UV" pip install --python "$P" torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  "$P" -c 'import triton' >/dev/null 2>&1 || "$UV" pip install --python "$P" triton==3.4.0
fi

export CC="${CC:-$(command -v gcc || true)}"
export CXX="${CXX:-$(command -v g++ || true)}"
if [ -z "$CC" ] || [ -z "$CXX" ]; then
  echo "Stage-5 requires gcc/g++ for TorchInductor" >&2
  exit 43
fi
export TORCHINDUCTOR_CACHE_DIR="$CACHE"
export PYTHONUNBUFFERED=1

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
"$P" -m py_compile frontier350/model350.py frontier350/model350_fast.py frontier350/speed_stage5_trainpath.py
"$P" - <<'PY'
import sys
sys.path.insert(0, 'frontier350')
from model350_fast import Neral350MFast
r = Neral350MFast().parameter_report()
print(r)
assert r['total'] == 349_825_152
PY

mkdir -p "$(dirname "$OUT")"
"$P" -u frontier350/speed_stage5_trainpath.py \
  --train-data "$TRAIN" \
  --out "$OUT" \
  --seq 512 \
  --updates 20 \
  --warmup 5
