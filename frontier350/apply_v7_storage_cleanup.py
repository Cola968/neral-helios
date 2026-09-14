from __future__ import annotations

"""Bound Scale V7 scratch usage after each 5M transaction.

Recovery files are useful only while the current stage is unresolved. Once a stage is
committed to the durable continuity journal, or rejected after all recovery probes fail,
its raw trainer directory and probe/interpolation files must not accumulate on D:.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "run_scale_v5.py"


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"V7 cleanup patch anchor mismatch for {label}: {count}")
    return text.replace(old, new, 1)


def patch_controller(text: str) -> str:
    text = once(
        text,
        "        recovery_states=list_recovery_states(out)\n        return raw, recovery_states, perf\n\n    def make_interp",
        "        recovery_states=list_recovery_states(out)\n        return raw, recovery_states, perf\n\n"
        "    def cleanup_v7_stage(tag: str) -> None:\n"
        "        # The accepted continuity checkpoint is moved to the durable journal before\n"
        "        # this is called. Everything still carrying the stage tag is disposable.\n"
        "        shutil.rmtree(scratch / tag, ignore_errors=True)\n"
        "        for p in scratch.glob(tag + '*'):\n"
        "            if p.is_file():\n"
        "                try:p.unlink()\n"
        "                except OSError:pass\n\n"
        "    def make_interp",
        "cleanup helper",
    )
    text = once(
        text,
        "            persist_state(\"STAGE_REJECTED\")\n            if consecutive_rejects >= 4:\n",
        "            persist_state(\"STAGE_REJECTED\")\n            cleanup_v7_stage(tag)\n            if consecutive_rejects >= 4:\n",
        "rejected stage cleanup",
    )
    text = once(
        text,
        "        accepted_stages += 1\n        preferred_alpha = max(0.006, min(0.10, float(accepted[\"alpha\"]) * 1.05))\n",
        "        accepted_stages += 1\n        cleanup_v7_stage(tag)\n        preferred_alpha = max(0.006, min(0.10, float(accepted[\"alpha\"]) * 1.05))\n",
        "committed stage cleanup",
    )
    return text


def main() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    CONTROLLER.write_text(patch_controller(text), encoding="utf-8")
    print("Scale V7 scratch cleanup applied: unresolved stage only; durable journal untouched")


if __name__ == "__main__":
    main()
