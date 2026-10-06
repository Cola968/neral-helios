"""Pure-Python guard for historical V7.1/V7.2 profile retries.

NEVER deletes, rewinds, authenticates or loads checkpoint files. A discovered
candidate must still pass the trainer's exact-recovery contract. When a failed
attempt left *anything* in the output directory and has no recovery candidate,
stop rather than silently discarding logs or replaying tokens from the anchor.
"""
from __future__ import annotations

from pathlib import Path


def fallback_allowed(output: Path, recovery: Path | None) -> bool:
    """Allow a retry only with a local recovery candidate or untouched output."""
    if output.is_symlink():
        return False
    if recovery is not None:
        return (output.is_dir() and recovery.parent == output
                and not recovery.is_symlink() and recovery.is_file()
                and recovery.stat().st_size > 0)
    if not output.exists():
        return True
    return output.is_dir() and not any(output.iterdir())
