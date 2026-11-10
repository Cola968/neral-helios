"""Operator entry point for the FUTURE approved bounded SFT pilot.

This does not grant training permission and is not called by GitHub Actions.
Require THREE independent stable GPU-idle measurements BEFORE the existing
pilot code may initialize Torch/CUDA. Preserve all pilot evidence and SHA gates.
Do not use this to bypass a missing holdout, source review, or operator approval.
"""
from __future__ import annotations

import sys

import phoenix_cuda_diagnostic as gpu_safety
import train_sft_pilot as pilot


def main() -> int:
    args = sys.argv[1:]
    if "--execute" not in args or "--trusted-local-checkpoint" not in args:
        raise SystemExit("Explicit --execute and --trusted-local-checkpoint are required; no GPU or training started")
    # Three snapshots, 2 seconds apart; ALL must satisfy utilization <=15%,
    # used VRAM <=2500 MiB and temperature <=78 C. This happens BEFORE
    # train_sft_pilot.main imports Torch or allocates CUDA memory.
    gpu_safety.require_stably_idle_gpu()
    return pilot.main()  # Real pack/evidence/checkpoint validation remains mandatory.


if __name__ == "__main__":
    raise SystemExit(main())
