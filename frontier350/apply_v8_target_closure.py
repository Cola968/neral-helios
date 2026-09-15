from __future__ import annotations

"""V8 additive controller hardening for long 2B->3B continuation.

Keeps V5/V6/V7/V7.2 training math, recipes, recovery and SAFE gates unchanged. It only
fixes run-completion semantics: a wall-limited partial run can no longer masquerade as a
finished target run, and a small final <=5M stage gets a bounded grace window so an exact
milestone is not abandoned a few minutes before completion.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "run_scale_v5.py"


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"V8 target-closure patch anchor mismatch for {label}: {count}")
    return text.replace(old, new, 1)


def patch_controller(text: str) -> str:
    text = once(
        text,
        '    ap.add_argument("--max-wall-minutes", type=int, default=4200)\n',
        '    ap.add_argument("--max-wall-minutes", type=int, default=4200)\n'
        '    ap.add_argument("--target-finish-grace-minutes", type=int, default=20)\n'
        '    ap.add_argument("--require-target", action="store_true")\n',
        "arguments",
    )
    text = once(
        text,
        '    a = ap.parse_args()\n\n    assets, report, scratch =',
        '    a = ap.parse_args()\n'
        '    if a.max_wall_minutes <= 0 or a.target_finish_grace_minutes < 0:\n'
        '        raise ValueError("invalid wall/grace budget")\n\n'
        '    assets, report, scratch =',
        "argument validation",
    )
    old_loop = '''    round_no = 0
    consecutive_rejects = 0
    while (train_tokens < a.target_tokens and round_no < a.max_rounds and
           (time.monotonic() - started) / 60 < a.max_wall_minutes):
'''
    new_loop = '''    def run_window_open() -> bool:
        elapsed = (time.monotonic() - started) / 60
        if elapsed < a.max_wall_minutes:
            return True
        remaining = a.target_tokens - train_tokens
        # Exact milestone closure: if only one normal micro-stage remains, allow a small
        # bounded grace window instead of reporting success a few million tokens early.
        return (0 < remaining <= STAGE_TOKENS and
                elapsed < a.max_wall_minutes + a.target_finish_grace_minutes)

    round_no = 0
    consecutive_rejects = 0
    while train_tokens < a.target_tokens and round_no < a.max_rounds and run_window_open():
'''
    text = once(text, old_loop, new_loop, "wall-aware loop")

    result_anchor = '''    result = {
        "schema": 6,
'''
    result_new = '''    elapsed_minutes = (time.monotonic() - started) / 60
    target_reached = train_tokens >= a.target_tokens
    if target_reached:
        stop_reason = "TARGET_REACHED"
    elif round_no >= a.max_rounds:
        stop_reason = "MAX_ROUNDS"
    elif elapsed_minutes >= a.max_wall_minutes:
        stop_reason = "WALL_LIMIT"
    else:
        stop_reason = "EARLY_STOP_OR_REJECT_LIMIT"

    result = {
        "schema": 7,
'''
    text = once(text, result_anchor, result_new, "completion classification")
    text = once(
        text,
        '        "target_tokens": a.target_tokens,\n',
        '        "target_tokens": a.target_tokens,\n'
        '        "target_reached": target_reached,\n'
        '        "remaining_tokens": max(0, a.target_tokens - train_tokens),\n'
        '        "completion_status": "TARGET_REACHED" if target_reached else "PARTIAL_PROGRESS",\n'
        '        "stop_reason": stop_reason,\n'
        '        "target_finish_grace_minutes": a.target_finish_grace_minutes,\n',
        "result target metadata",
    )
    text = once(
        text,
        '        "wall_minutes": (time.monotonic() - started) / 60,\n',
        '        "wall_minutes": elapsed_minutes,\n',
        "stable wall metric",
    )
    final_old = '''    print(json.dumps({"event": "final_selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
'''
    final_new = '''    print(json.dumps({"event": "final_selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    if a.require_target and not target_reached:
        raise RuntimeError(
            f"required target not reached: {train_tokens:,}/{a.target_tokens:,} tokens; "
            f"remaining={a.target_tokens-train_tokens:,}; reason={stop_reason}"
        )
'''
    text = once(text, final_old, final_new, "require-target failure")
    return text


def main() -> None:
    original = CONTROLLER.read_text(encoding="utf-8")
    patched = patch_controller(original)
    if patched != original:
        CONTROLLER.write_text(patched, encoding="utf-8")
    print(
        "Scale V8 target closure applied: bounded final-stage grace + explicit partial/target status + require-target contract"
    )


if __name__ == "__main__":
    main()
