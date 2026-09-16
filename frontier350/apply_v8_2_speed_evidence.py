from __future__ import annotations

"""Add passive, monotonic stage wall-time evidence to the V8 controller.

Apply AFTER the existing V6/V7/V8 patch chain. Only progress.json metadata
changes: never optimizer, inputs, quality screens, checkpoints or thermal guards.
"""

from pathlib import Path

CONTROLLER = Path(__file__).resolve().parent / "run_scale_v5.py"


def patch_controller(text: str) -> str:
    start_old = "        round_no += 1\n        before_tokens = train_tokens\n"
    start_new = (
        "        round_no += 1\n"
        "        stage_started_elapsed_seconds = time.monotonic() - started\n"
        "        before_tokens = train_tokens\n"
    )
    reject_old = 'history.append({"round": round_no, "status": "STAGE_REJECTED_NO_REWIND",\n'
    reject_new = (
        'history.append({"round": round_no, "status": "STAGE_REJECTED_NO_REWIND",\n'
        '                            "stage_start_elapsed_seconds": stage_started_elapsed_seconds,\n'
        '                            "stage_end_elapsed_seconds": time.monotonic() - started,\n'
    )
    accept_old = 'history.append({"round": round_no, "status": "CONTINUITY_COMMIT",\n'
    accept_new = (
        'history.append({"round": round_no, "status": "CONTINUITY_COMMIT",\n'
        '                        "stage_start_elapsed_seconds": stage_started_elapsed_seconds,\n'
        '                        "stage_end_elapsed_seconds": time.monotonic() - started,\n'
    )
    for old, new in ((start_old, start_new), (reject_old, reject_new), (accept_old, accept_new)):
        if new in text:
            continue
        if text.count(old) != 1:
            raise RuntimeError(f"V8.2 performance evidence patch anchor mismatch: {old!r}")
        text = text.replace(old, new, 1)
    return text


def main() -> None:
    original = CONTROLLER.read_text(encoding="utf-8")
    patched = patch_controller(original)
    if patched != original:
        CONTROLLER.write_text(patched, encoding="utf-8")
    print("V8.2: wall-time evidence enabled; optimizer and quality gates unchanged")


if __name__ == "__main__":
    main()
