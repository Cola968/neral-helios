"""Explicit GPU Lab opt-in must never silently alter the ordinary CPU starter."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lab_launcher_is_separate_from_ordinary_cpu_start():
    normal = (ROOT / "Start Neral.cmd").read_text(encoding="utf-8")
    lab = (ROOT / "Start NERAL v5 Lab.cmd").read_text(encoding="utf-8")
    assert "-GpuLab" in lab
    assert "-GpuLab" not in normal
    assert "Start-Neral.ps1" in lab


def test_explicit_lab_uses_existing_gpu_python_and_never_downloads_weights():
    source = (ROOT / "Start-Neral.ps1").read_text(encoding="utf-8")
    assert "[switch]$GpuLab" in source
    assert "$env:NERAL_AGENT_BACKEND='hf_adapter'" in source
    assert "$env:NERAL_AGENT_ADAPTER_CHANNEL='v5'" in source
    assert "training-env-py312" in source
    assert "$env:CUDA_VISIBLE_DEVICES='0'" in source
    assert "$env:CUDA_VISIBLE_DEVICES=''" in source
    assert "NERAL v5" in source
    assert "already running" in source.lower() or "bereits" in source.lower()


def test_regular_start_preserves_disabled_gpu_default():
    source = (ROOT / "Start-Neral.ps1").read_text(encoding="utf-8")
    gpu_placement = source.index("$env:CUDA_VISIBLE_DEVICES='0'")
    cpu_placement = source.index("$env:CUDA_VISIBLE_DEVICES=''")
    assert gpu_placement < cpu_placement
    assert "if($GpuLab)" in source
