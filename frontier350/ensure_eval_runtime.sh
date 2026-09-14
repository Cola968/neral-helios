#!/usr/bin/env bash
# Keep this file LF-only; it is executed inside WSL from a Windows runner.
set -euo pipefail

P="${HOME}/.venvs/neral-triton-py312/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${HERE}/apply_scale_v4_recovery_patch.py"

if [[ ! -x "$P" ]]; then
  echo "missing WSL eval python: $P" >&2
  exit 1
fi

# Future Scale V4 runs patch the checked-out trainer/controller before the workflow's
# existing py_compile gate. The currently running job is pinned to its older commit and
# therefore remains untouched.
if [[ -f "$PATCH" ]]; then
  "$P" "$PATCH"
fi

if ! "$P" -c 'import tokenizers; assert tokenizers.__version__ == "0.21.4"' >/dev/null 2>&1; then
  "$P" -m ensurepip --upgrade >/dev/null
  "$P" -m pip install --disable-pip-version-check 'tokenizers==0.21.4'
fi

"$P" -c 'import torch, tokenizers; assert tokenizers.__version__ == "0.21.4"; print("torch", torch.__version__, "tokenizers", tokenizers.__version__, "cuda", torch.cuda.is_available())'