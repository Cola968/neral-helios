from __future__ import annotations

"""Route Scale V8.2 through Fast2 + admitted coalesced-input execution.

Only the stage launcher changes. Training data, RNG/token order, optimizer math, LR/replay
recipe, recovery cadence, health gates and strict SAFE audits remain owned by the existing
V5/V6/V7 controller/trainer chain.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "run_scale_v5.py"


def patch_controller(text: str) -> str:
    new = 'str(Path(__file__).parent / "train_v8_2_stage_launcher.py")'
    if new in text:
        return text
    old = 'str(Path(__file__).parent / "train_v7_2_stage_launcher.py")'
    if text.count(old) != 1:
        raise RuntimeError(f"V8.2 launcher anchor mismatch: {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    CONTROLLER.write_text(patch_controller(text), encoding="utf-8")
    print("Scale V8.2 applied: Fast2+coalesced first, plain Fast2 and V7 fallbacks preserved")


if __name__ == "__main__":
    main()
