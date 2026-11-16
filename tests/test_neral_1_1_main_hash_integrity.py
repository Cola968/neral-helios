"""CPU evidence against false-positive Windows CRLF drift and real Git blob changes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import subprocess

import pytest
from scripts import neral_1_1_main_cpu_gate as gate


def test_normal_windows_crlf_worktree_is_accepted():
    old = Path.read_bytes
    chosen = (gate.ROOT / gate.PROTECTED[0]).resolve()
    def windows_bytes(self):
        value = old(self)
        if self.resolve() == chosen:
            return value.replace(b"\\n", b"\\r\\n")
        return value
    with patch.object(Path, "read_bytes", windows_bytes):
        gate.verify_main_integrity()


def test_pinned_main_blob_mismatch_is_rejected():
    original = subprocess.run
    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"] and cmd[2] == "HEAD:"+gate.PROTECTED[0]:
            return subprocess.CompletedProcess(cmd, 0, stdout="f"*40+"\\n")
        return original(cmd, **kwargs)
    with patch.object(subprocess, "run", fake_run):
        with pytest.raises(RuntimeError, match="CURRENT_MAIN_WEB_OR_UI_CHANGED"):
            gate.verify_main_integrity()


def test_modified_worktree_is_rejected():
    original = subprocess.run
    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["git", "diff", "--quiet"] and gate.PROTECTED[0] in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="")
        return original(cmd, **kwargs)
    with patch.object(subprocess, "run", fake_run):
        with pytest.raises(RuntimeError, match="WORKTREE_DRIFT"):
            gate.verify_main_integrity()
