from __future__ import annotations

"""Production continuation wrapper for Scale V3 after the bounded Pack9 canary.

The underlying controller, optimizer, replay policy, trainable layers, token accounting,
and promotion gates remain unchanged. Only physical main-microbatch grouping is changed
from Pack6 to Pack9. Runtime preflight in the workflow requires the Pack9 canary's
promoted safe anchor to be present before this wrapper is allowed to train.
"""

import run_scale_v3_success as scale


def main() -> None:
    if scale.PACK != 6:
        raise RuntimeError(f"unexpected Scale V3 baseline PACK={scale.PACK}; review required")
    scale.PACK = 9
    scale.main()


if __name__ == "__main__":
    main()
