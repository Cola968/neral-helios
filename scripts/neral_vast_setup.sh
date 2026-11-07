#!/usr/bin/env bash
# Manual setup INSIDE an already-rented Vast.ai PyTorch Linux container.
# No GPU rental, training, model upload, or host-OS modifications.
set -euo pipefail
if [[ "$(uname -s)" != "Linux" ]]; then echo "Linux only" >&2; exit 2; fi
if [[ "$(uname -m)" != "x86_64" ]]; then echo "x86_64 required, not ARM" >&2; exit 2; fi
if [[ ! -d /workspace ]]; then echo "/workspace not found; check Vast disk" >&2; exit 2; fi
command -v git >/dev/null || { echo "git missing" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "GPU driver unavailable" >&2; exit 2; }
python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Python 3.10-3.12 required"'
REPO=/workspace/neral-helios
if [[ ! -f "$REPO/scripts/neral_vast_cloud.py" ]]; then
  echo "Checkout NERAL repository into /workspace/neral-helios first" >&2; exit 2
fi
PINNED=/workspace/neral-cloud-pinned
REV=a082738f9145722dec443141e5d7801a3ead93af
if [[ ! -d "$PINNED/.git" ]]; then
  if [[ -e "$PINNED" ]]; then echo "Unverified pinned path" >&2; exit 2; fi
  git clone --filter=blob:none --no-checkout https://github.com/Cola968/neral-helios.git "$PINNED"
fi
git -C "$PINNED" fetch --depth=1 origin "$REV"
git -C "$PINNED" checkout --detach "$REV"
if [[ "$(git -C "$PINNED" rev-parse HEAD)" != "$REV" ]]; then
  echo "Pinned trainer source differs" >&2; exit 2
fi
if [[ ! -f "$PINNED/requirements-neral-training.txt" ]]; then
  echo "Pinned QLoRA dependencies not found" >&2; exit 2
fi
ENV=/workspace/neral-cloud-env
if [[ ! -x "$ENV/bin/python" ]]; then python3 -m venv "$ENV"; fi
"$ENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip
"$ENV/bin/python" -m pip install --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0+cu128'
"$ENV/bin/python" -m pip install --disable-pip-version-check \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "$PINNED/requirements-neral-training.txt"
cd "$REPO"
"$ENV/bin/python" -B -m unittest discover -s tests -p test_neral_vast_cloud.py -v
"$ENV/bin/python" -B -m scripts.neral_vast_cloud preflight
echo "NERAL_CLOUD_ENV_READY; NO TRAINING AND NO MODEL DOWNLOAD"
