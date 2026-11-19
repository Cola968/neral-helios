"""Bounded offline Docker daemon readiness for NERAL's local sandbox.

Never installs software, pulls an image, changes context or executes model code
outside the pinned Docker container. Starts only the known user-scoped Desktop
executable on Windows when the daemon is unavailable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import threading
import time

_LOG = logging.getLogger("neral.docker")
_LOCK = threading.Lock()


def _daemon_ready() -> bool:
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, check=False, timeout=4)
        return info.returncode == 0 and bool(info.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def _launch_existing_desktop() -> bool:
    if os.name != "nt":
        return False
    root = os.environ.get("LOCALAPPDATA")
    if not root:
        return False
    exe = Path(root) / "Programs" / "DockerDesktop" / "Docker Desktop.exe"
    if not exe.is_file() or exe.is_symlink():
        _LOG.warning("NERAL_DOCKER_DESKTOP_PINNED_EXE_ABSENT")
        return False
    try:
        subprocess.Popen(
            [str(exe)], cwd=str(exe.parent),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True,
            creationflags=(subprocess.DETACHED_PROCESS |
                           subprocess.CREATE_NEW_PROCESS_GROUP))
    except (OSError, ValueError):
        _LOG.warning("NERAL_DOCKER_DESKTOP_START_FAILED")
        return False
    _LOG.info("NERAL_DOCKER_DESKTOP_START_REQUESTED")
    return True


def ensure_docker_ready() -> bool:
    """Check daemon before each execution; cold-start existing Desktop once.

    The lock prevents duplicate startups by concurrent requests. Never replaces
    the sandbox's no-network, read-only execution limits.
    """
    if _daemon_ready():
        return True
    _LOG.warning("NERAL_DOCKER_DAEMON_UNAVAILABLE")
    if os.name != "nt":
        return False
    with _LOCK:
        if _daemon_ready():
            return True
        if not _launch_existing_desktop():
            return False
        for attempt in range(12):
            time.sleep(2)
            if _daemon_ready():
                _LOG.info("NERAL_DOCKER_DAEMON_READY attempt=%d", attempt+1)
                return True
    _LOG.warning("NERAL_DOCKER_DAEMON_TIMEOUT")
    return False
