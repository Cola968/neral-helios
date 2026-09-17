"""Read-only capacity gate for an opt-in checkpoint run; never deletes files.

Hard-linked generations are charged once by filesystem identity. A journal with
unbounded retention must STOP at the budget, not prune protected checkpoints.
This gate covers the participating training directory, not all legacy workflows.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil

GIB = 1024 ** 3
MAX_DIRECTORY_BYTES = 40 * GIB
MIN_FREE_BYTES = 200 * GIB
EXTRA_RESERVE_BYTES = GIB


def _no_redirect(path: Path) -> None:
    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        raise ValueError('storage budget refuses symbolic links and junctions inside training directory')


def physical_directory_bytes(directory: Path) -> int:
    """Conservative physical-size estimate, deduplicating only verified hard links."""
    directory = Path(directory).absolute()
    if not directory.is_dir():
        raise FileNotFoundError('training directory absent')
    _no_redirect(directory)
    seen: set[tuple[int, int] | tuple[str, str]] = set()
    total = 0

    def error(exc: OSError) -> None:
        raise exc

    for parent, folders, files in os.walk(directory, followlinks=False, onerror=error):
        for name in folders:
            _no_redirect(Path(parent) / name)
        for name in files:
            path = Path(parent) / name
            _no_redirect(path)
            st = path.stat()
            if not path.is_file():
                raise ValueError('non-regular file inside checkpoint directory')
            identity = (st.st_dev, st.st_ino) if st.st_ino else ('path', str(path))
            if identity not in seen:
                seen.add(identity)
                total += st.st_size
    return total


def check_storage_budget(directory: Path, resume: Path, *,
                         max_directory_bytes: int = MAX_DIRECTORY_BYTES,
                         min_free_bytes: int = MIN_FREE_BYTES) -> dict[str, int]:
    """Reserve space for a new atomic checkpoint BEFORE a model/GPU write.

    The existing checkpoint remains hard-linked by the journal, so a new write
    may require a whole additional checkpoint. Reserve twice its current size
    plus 1 GiB for temporary overhead. Exact prediction is impossible; the
    post-save check must still stop on unexpected growth.
    """
    directory = Path(directory).absolute()
    resume = Path(resume).absolute()
    if not 8 * GIB <= max_directory_bytes <= MAX_DIRECTORY_BYTES:
        raise ValueError('directory budget must be 8..40 GiB')
    if min_free_bytes < MIN_FREE_BYTES:
        raise ValueError('minimum free-space reserve cannot be relaxed below 200 GiB')
    if resume.parent != directory or resume.name != 'resume.pt':
        raise ValueError('budget requires resume.pt directly inside training directory')
    _no_redirect(resume)
    if not resume.is_file():
        raise FileNotFoundError('resumable checkpoint absent')
    checkpoint_bytes = resume.stat().st_size
    if checkpoint_bytes <= 0:
        raise ValueError('empty checkpoint is not a valid storage estimate')
    used = physical_directory_bytes(directory)
    anticipated = checkpoint_bytes * 2 + EXTRA_RESERVE_BYTES
    free = shutil.disk_usage(directory).free
    if used + anticipated > max_directory_bytes:
        raise RuntimeError('training directory would exceed 40 GiB safety budget; no files deleted')
    if free - anticipated < min_free_bytes:
        raise RuntimeError('disk would fall below 200 GiB free-space reserve; no files deleted')
    return {'directory_bytes': used, 'free_bytes': free,
            'reserved_write_bytes': anticipated, 'directory_limit_bytes': max_directory_bytes,
            'minimum_free_bytes': min_free_bytes}
