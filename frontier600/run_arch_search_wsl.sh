#!/usr/bin/env bash
set -euo pipefail

REPO=${1:?repo path required}
TRAIN=${2:?train-data path required}
VALID=${3:?valid-data path required}
OUT=${4:?output path required}
VENV="$HOME/.venvs/neral-triton-py312"
BOOT="$HOME/.venvs/neral-uv-bootstrap"
CACHE="$HOME/.cache/neral_inductor_600"

mkdir -p "$HOME/.venvs" "$CACHE" "$OUT"

# Reuse the known-good Neral WSL2 environment contract used by Arctus Stage-5.
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
  echo "Neral600 search requires gcc/g++ for TorchInductor" >&2
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
print('vram_GiB', torch.cuda.get_device_properties(0).total_memory / 2**30)
assert sys.version_info[:2] == (3, 12)
assert torch.__version__.startswith('2.8.')
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
PY

cd "$REPO"
test -d "$TRAIN"
test -d "$VALID"
"$P" -m py_compile \
  frontier600/architecture_search600.py \
  frontier600/model600.py \
  frontier600/bench600.py \
  frontier600/train_pilot600.py \
  frontier600/score600.py

# Phase A/B: exact analytical contract. Fails immediately if any candidate is out of range.
PYTHONPATH=frontier600 "$P" frontier600/architecture_search600.py > "$OUT/phase_ab_analytical.json"

# Runtime construction gate: exact torch parameter count must equal the analytical count.
PYTHONPATH=frontier600 "$P" - <<'PY' > "$OUT/model_integrity.json"
import json, gc, torch
from architecture_search600 import PHASE_C, CANDIDATES
from model600 import make_model
rows=[]
for cid in PHASE_C:
    m=make_model(cid)
    expected=CANDIDATES[cid].parameter_breakdown()['total']
    actual=sum(p.numel() for p in m.parameters())
    tied=m.lm_head.weight is m.tok_emb.weight
    rows.append({'candidate':cid,'expected':expected,'actual':actual,'tied':tied})
    assert actual==expected and tied
    del m; gc.collect()
print(json.dumps(rows, indent=2))
PY

# Phase C: real RTX-4070 optimizer-inclusive speed/VRAM gate.
PYTHONPATH=frontier600 "$P" -u frontier600/bench600.py \
  --candidates w26 b32 d36 q36 \
  --seq 512 \
  --microbatches 1 2 \
  --tokens-per-update 16384 \
  --warmup 2 \
  --measured 4 \
  --compile-mode default \
  --out "$OUT/phase_c_speed.json"

# Phase D: matched 5M-token learning curves. This is still only a short pilot.
PYTHONPATH=frontier600 "$P" -u frontier600/train_pilot600.py \
  --train-data "$TRAIN" \
  --valid-data "$VALID" \
  --candidates w26 b32 d36 q36 \
  --out "$OUT/phase_d_5m" \
  --target-tokens 5000000 \
  --seq 512 \
  --microbatch 1 \
  --tokens-per-update 16384 \
  --checkpoint-every 2 \
  --compile-mode default \
  --lr 3e-4 \
  --warmup-tokens 500000 \
  --eval-every 50 \
  --eval-batches 12 \
  --log-every 10 \
  --seed 600001

# Phase E: weighted evidence-gated ranking. It remains provisional after 5M tokens.
PYTHONPATH=frontier600 "$P" -u frontier600/score600.py \
  --speed "$OUT/phase_c_speed.json" \
  --pilots "$OUT/phase_d_5m/all_summaries.json" \
  --out "$OUT/phase_e_score.json"

printf '\nNeral600 architecture search completed. Reports: %s\n' "$OUT"
