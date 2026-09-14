#!/usr/bin/env bash
# Keep this file LF-only; it is executed inside WSL from a Windows runner.
set -euo pipefail

P="${HOME}/.venvs/neral-triton-py312/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${HERE}/apply_scale_v4_recovery_patch.py"
ROLLBACK_CAP="${HERE}/../sitecustomize.py"
V5_TURBO="${HERE}/apply_v5_turbo_patch.py"
V5_LAUNCHER="${HERE}/train_v5_stage_launcher.py"

if [[ ! -x "$P" ]]; then
  echo "missing WSL eval python: $P" >&2
  exit 1
fi

# Future Scale V4/V5 runs patch the checked-out trainer/controller before the workflow's
# existing py_compile gate. A job already pinned to an older commit remains untouched.
if [[ -f "$PATCH" ]]; then
  "$P" "$PATCH"
fi

# Preserve the strict 5M rollback cap for V4 compatibility.
if [[ -f "$ROLLBACK_CAP" ]]; then
  "$P" "$ROLLBACK_CAP"
fi

# V5 keeps the same 5M continuity policy but uses an execution-only turbo ladder:
# turbo60 = fused replay+main + max-autotune/cudagraphs; turbo50 = fused +
# max-autotune-no-cudagraphs; safe = legacy fast replay_only/default compile.
# A failed fast profile automatically retries the same stage with the safer profile.
if [[ -f "$V5_TURBO" ]]; then
  "$P" "$V5_TURBO"
  "$P" -m py_compile "$V5_TURBO" "$V5_LAUNCHER" "$HERE/run_scale_v5.py" "$HERE/train_guarded_v9.py"
fi

if ! "$P" -c 'import tokenizers; assert tokenizers.__version__ == "0.21.4"' >/dev/null 2>&1; then
  "$P" -m ensurepip --upgrade >/dev/null
  "$P" -m pip install --disable-pip-version-check 'tokenizers==0.21.4'
fi

"$P" -c 'import torch, tokenizers; assert tokenizers.__version__ == "0.21.4"; print("torch", torch.__version__, "tokenizers", tokenizers.__version__, "cuda", torch.cuda.is_available())'
