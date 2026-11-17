"""Hermetic CPU security contracts for pinned main integrity without shallow Git history.

Production gate remains fail-closed and unchanged. GitHub Actions checkout
with fetch-depth=1 cannot resolve its historical protected BASE blob, so
mock only immutable git process outputs and assert the full security behavior.
"""
from pathlib import Path
from unittest.mock import patch
import subprocess
import pytest
from scripts import neral_1_1_main_cpu_gate as gate


def _git(case="same"):
    calls=[]
    def run(cmd,**kwargs):
        calls.append(tuple(cmd))
        if cmd[:2]==["git","rev-parse"]:
            assert cmd[2].startswith(("HEAD:","2de4b034d79521e4f5c9d432af7f98a70087ed70:"))
            if case=="mismatch" and cmd[2].startswith("HEAD:"):
                digest="f"*40
            else:
                digest="a"*40
            return subprocess.CompletedProcess(cmd,0,stdout=digest+"\n")
        if cmd[:3]==["git","diff","--quiet"]:
            assert cmd[3:5]==["HEAD","--"]
            return subprocess.CompletedProcess(cmd,1 if case=="dirty" else 0,stdout="")
        raise AssertionError("Unexpected git operation: "+repr(cmd))
    return run,calls


def test_normal_windows_crlf_worktree_is_accepted():
    run,calls=_git()
    # Gate intentionally relies on git blob identity; checked-out CRLF
    # transformations are not compared as raw bytes.
    with patch.object(subprocess,"run",run):
        gate.verify_main_integrity()
    assert len([c for c in calls if c[:2]==("git","rev-parse")])==2*len(gate.PROTECTED)
    assert len([c for c in calls if c[:3]==("git","diff","--quiet")])==len(gate.PROTECTED)


def test_pinned_main_blob_mismatch_is_rejected():
    run,calls=_git("mismatch")
    with patch.object(subprocess,"run",run):
        with pytest.raises(RuntimeError,match="CURRENT_MAIN_WEB_OR_UI_CHANGED"):
            gate.verify_main_integrity()


def test_modified_worktree_is_rejected():
    run,calls=_git("dirty")
    with patch.object(subprocess,"run",run):
        with pytest.raises(RuntimeError,match="WORKTREE_DRIFT"):
            gate.verify_main_integrity()
