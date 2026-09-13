from __future__ import annotations

"""Bounded Pack9 qualification wrapper for Scale V3.

This changes only the physical grouping of the nine main microbatches. Token accounting,
replay mix, optimizer, trainable layers, recovery logic, and all promotion gates remain
owned by run_scale_v3_success.py and are unchanged.
"""

import run_scale_v3_success as scale


def main() -> None:
    if scale.PACK != 6:
        raise RuntimeError(f"unexpected Scale V3 baseline PACK={scale.PACK}; review required")
    scale.PACK = 9
    scale.main()


if __name__ == "__main__":
    main()
