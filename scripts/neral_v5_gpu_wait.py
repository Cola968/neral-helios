"""Fail-closed bounded GPU readiness check for NERAL's read-only A/B evaluation.

A previous 320-case Qwen evaluation succeeded but the following NERAL variant
could not start: <9000 MiB free or >20% utilization immediately after Qwen.
This script retries for at most 180 seconds without weakening those gates.
No model is loaded here; this script does not free or terminate processes.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections.abc import Callable

MIN_FREE_MIB = 9000
MAX_GPU_UTIL_PCT = 20
MAX_CHECKS = 19
POLL_SECONDS = 10


class GPUReadyError(RuntimeError):
    """The GPU is unavailable or cannot safely be inspected."""


def parse_gpu_csv(raw: str) -> tuple[int, int]:
    """One NVIDIA GPU, two numeric counters; reject partial/ambiguous output."""
    if not isinstance(raw, str):
        raise GPUReadyError("GPU sensor returned non-text output")
    lines = raw.strip().splitlines()
    if len(lines) != 1:
        raise GPUReadyError("GPU sensor must return exactly one NVIDIA GPU")
    match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", lines[0])
    if match is None:
        raise GPUReadyError("GPU sensor returned malformed counters")
    free_mib, utilization = map(int, match.groups())
    if not (0 <= free_mib <= 2_000_000 and 0 <= utilization <= 100):
        raise GPUReadyError("GPU sensor returned out-of-range counters")
    return free_mib, utilization


def read_nvidia_gpu() -> tuple[int, int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GPUReadyError("GPU sensor unavailable") from exc
    if result.returncode != 0:
        raise GPUReadyError("GPU sensor command failed")
    return parse_gpu_csv(result.stdout)


def wait_until_gpu_ready(
    variant: str,
    read: Callable[[], tuple[int, int]],
    sleep: Callable[[int], None],
    log: Callable[[str], None],
) -> tuple[int, int]:
    """Retry 18 times; never proceed with insufficient memory or busy GPU."""
    if variant not in ("base", "neral_v5"):
        raise GPUReadyError("Unexpected model variant")
    for attempt in range(1, MAX_CHECKS + 1):
        measurement = read()
        if (not isinstance(measurement, tuple) or len(measurement) != 2
                or any(type(v) is not int for v in measurement)
                or measurement[0] < 0 or measurement[1] not in range(101)):
            raise GPUReadyError("Invalid GPU sensor measurement")
        free_mib, utilization = measurement
        if free_mib >= MIN_FREE_MIB and utilization <= MAX_GPU_UTIL_PCT:
            log(f"GPU ready for {variant}: free={free_mib} MiB, "
                f"utilization={utilization}% (check {attempt}/{MAX_CHECKS})")
            return measurement
        if attempt == MAX_CHECKS:
            break
        log(f"GPU not ready for {variant}: free={free_mib} MiB "
            f"(required {MIN_FREE_MIB}), utilization={utilization}% "
            f"(maximum {MAX_GPU_UTIL_PCT}); waiting {POLL_SECONDS}s "
            f"(check {attempt}/{MAX_CHECKS})")
        sleep(POLL_SECONDS)
    raise GPUReadyError(
        f"GPU still busy before {variant} after {MAX_CHECKS} checks "
        f"({(MAX_CHECKS - 1) * POLL_SECONDS}s); no memory limit was relaxed"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--variant", choices=("base", "neral_v5"), required=True)
    args = parser.parse_args(argv)
    try:
        wait_until_gpu_ready(args.variant, read_nvidia_gpu, time.sleep,
                             lambda msg: print(msg, flush=True))
    except GPUReadyError as exc:
        print(f"GPU readiness check failed: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
