"""Guard Windows WDDM graphics without allowing concurrent CUDA training."""
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/neral-v2-24-checkpoint-fast-gpu.yml"

def _gpu_gate() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("      - name: Verify RTX4070 GPU compute availability")
    end = text.index("      - name: Paired 24 real full NERAL pipeline", start)
    return text[start:end]

def test_wddm_desktop_processes_are_not_incorrectly_considered_compute_jobs():
    gate = _gpu_gate()
    assert "Get-Process -Id $gpuPid" in gate
    assert "GPU compute worker detected; preserving active workload" in gate
    assert "WINDOWS_WDDM_DESKTOP_PROCESSES_TOLERATED" in gate
    assert "Stop-Process" not in gate
    assert "taskkill" not in gate
    assert "ollama stop" not in gate

def test_gate_requires_sufficient_stable_vram_and_no_loaded_ollama():
    gate = _gpu_gate()
    assert "http://127.0.0.1:11434/api/ps" in gate
    assert "Ollama model already loaded" in gate
    assert "nvidia-smi --query-gpu=name,memory.total,memory.free" in gate
    assert "9500" in gate
    assert "DEV24_FAST_GPU_READY" in gate
    assert "Start-Sleep -Seconds 3" in gate

def test_immutable_checkpoint_and_branch_guards_preserved():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "dev24-hf-sourcefirst-bound256-v2-20261117.auto-start-20261117.json" in text
    assert "[IO.FileMode]::CreateNew" in text
    assert "--checkpoint-root $env:D24_CKPT" in text
    assert "runs-on: [self-hosted,windows,x64,neral-gpu]" in text
    assert "github.ref == 'refs/heads/feat/qlora-v2-full-and-luna'" in text
