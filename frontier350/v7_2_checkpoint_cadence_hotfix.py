from __future__ import annotations

"""V7.2-only checkpoint cadence fix.

V4 defines --resume-save-interval-tokens=0 as the legacy full-resume cadence
(every 20 updates). V7 intended compact 1M exact recovery states to replace
those interval full checkpoints, but passed 0 and therefore accidentally kept
legacy full saves alive.

For V7.2, keep the exact 1M recovery cadence unchanged and request one full
resume only at the 5M stage endpoint. V5/V6 source and fallback behavior are
not changed.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "run_scale_v5.py"

OLD = (
    '               "--snapshot-interval-tokens", "0",\n'
    '               "--resume-save-interval-tokens", "0",\n'
    '               "--v7-recovery-interval-tokens", str(V7_RECOVERY_INTERVAL_TOKENS)]\n'
)
NEW = (
    '               "--snapshot-interval-tokens", "0",\n'
    '               "--resume-save-interval-tokens", str(STAGE_TOKENS),\n'
    '               "--v7-recovery-interval-tokens", str(V7_RECOVERY_INTERVAL_TOKENS)]\n'
)


def patch_controller(text: str) -> str:
    if NEW in text:
        return text
    if text.count(OLD) != 1:
        raise RuntimeError(f"V7.2 checkpoint cadence anchor mismatch: {text.count(OLD)}")
    if "STAGE_TOKENS = 5_000_000" not in text:
        raise RuntimeError("unexpected stage token contract")
    if "V7_RECOVERY_INTERVAL_TOKENS = 1_000_000" not in text:
        raise RuntimeError("unexpected V7 recovery cadence")
    return text.replace(OLD, NEW, 1)


def main() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    patched = patch_controller(source)
    if patched != source:
        CONTROLLER.write_text(patched, encoding="utf-8")
    print(
        "V7.2 checkpoint cadence fixed: 1M exact recovery retained; "
        "legacy 20-update full saves disabled; full resume only at 5M endpoint"
    )


if __name__ == "__main__":
    main()
