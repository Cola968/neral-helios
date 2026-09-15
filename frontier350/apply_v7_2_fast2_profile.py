from __future__ import annotations

"""Route Scale V7.2 stages through the optional Fast2 execution launcher.

This changes only the execution launcher selected by the controller.  Stage size, quality
gates, recovery cadence, optimizer/data semantics and the proven V5/V6/V7 fallbacks stay
unchanged.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "run_scale_v5.py"


def patch_controller(text: str) -> str:
    new = 'str(Path(__file__).parent / "train_v7_2_stage_launcher.py")'
    if new in text:
        return text
    old = 'str(Path(__file__).parent / "train_v7_1_stage_launcher.py")'
    if text.count(old) != 1:
        raise RuntimeError(f"V7.2 launcher anchor mismatch: {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    CONTROLLER.write_text(patch_controller(text), encoding="utf-8")
    print("Scale V7.2 applied: gated Fast2 profile -> V7.1/V5 fallback chain")


if __name__ == "__main__":
    main()
