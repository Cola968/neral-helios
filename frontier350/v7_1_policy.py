from __future__ import annotations

"""Pure helpers for Scale V7.1 execution-only optimizations.

V7.1 must not change model/data/optimizer semantics.  These helpers make the batch-chain
rewrite and input-path admission independently testable on CPU.
"""

import hashlib
from typing import Any


def update_batch_chain(previous: str, domain: str, count: int, rows: Any) -> str:
    """Streaming equivalent of sha256(prev_bytes + domain + count + rows.tobytes())."""

    if len(previous) != 64:
        raise ValueError("previous batch-chain digest must be sha256 hex")
    if count <= 0:
        raise ValueError("token count must be positive")
    h = hashlib.sha256()
    h.update(bytes.fromhex(previous))
    h.update(domain.encode())
    h.update(int(count).to_bytes(8, "little"))
    h.update(memoryview(rows).cast("B"))
    return h.hexdigest()


def choose_io_profile(
    report: dict,
    *,
    min_speedup: float = 1.05,
    max_extra_reserved_gib: float = 0.25,
) -> tuple[str, dict]:
    """Choose an exact V7.1 input path or retain the V7 legacy path.

    Candidate admission requires byte-for-byte x/y equality, a measured speed win, and no
    meaningful reserved-VRAM regression.  Returning ``off`` means V7 behavior is retained.
    """

    legacy = report.get("legacy_split") or {}
    legacy_us = float(legacy.get("median_us", float("inf")))
    legacy_mem = float(legacy.get("peak_reserved_gib", float("inf")))
    candidates = []
    for name in ("coalesced-pin", "coalesced-pageable"):
        row = report.get(name.replace("-", "_")) or {}
        exact = bool(row.get("exact", False))
        us = float(row.get("median_us", float("inf")))
        mem = float(row.get("peak_reserved_gib", float("inf")))
        speedup = legacy_us / us if us > 0 and legacy_us < float("inf") else 0.0
        admitted = exact and speedup >= min_speedup and mem <= legacy_mem + max_extra_reserved_gib
        candidates.append({
            "profile": name,
            "exact": exact,
            "median_us": us,
            "peak_reserved_gib": mem,
            "speedup": speedup,
            "admitted": admitted,
        })
    eligible = [x for x in candidates if x["admitted"]]
    selected = max(eligible, key=lambda x: x["speedup"])["profile"] if eligible else "off"
    return selected, {
        "selected_profile": selected,
        "min_speedup": float(min_speedup),
        "max_extra_reserved_gib": float(max_extra_reserved_gib),
        "candidates": candidates,
    }
