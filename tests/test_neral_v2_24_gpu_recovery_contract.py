"""GPU recovery contract: never interrupt an unknown process to run DEV24."""
from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github/workflows/neral-v2-24-checkpoint-fast-gpu.yml"
)

def _gpu_gate() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("      - name: Require idle RTX4070")
    end = text.index("      - name: Paired 24 real full NERAL pipeline", start)
    return text[start:end]

def test_unknown_gpu_processes_must_abort_without_stopping_anything():
    gate = _gpu_gate()
    assert "Get-Process -Id $gpuPid" in gate
    assert "GPU occupied by non-Ollama process" in gate
    assert gate.index("GPU occupied by non-Ollama process") < gate.index("ollama stop")
    assert "Stop-Process" not in gate
    assert "taskkill" not in gate

def test_ollama_only_release_rechecks_gpu_and_vram_before_start():
    gate = _gpu_gate()
    assert "http://127.0.0.1:11434/api/ps" in gate
    assert "ollama stop" in gate
    assert "GPU still occupied after Ollama release" in gate
    assert "DEV24_FAST_GPU_READY" in gate
    assert "9500" in gate

def test_durable_case_checkpoint_claim_unchanged():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "dev24-hf-bound256-v1.auto-start-20261117.json" in text
    assert "[IO.FileMode]::CreateNew" in text
    assert "--checkpoint-root $env:D24_CKPT" in text
    assert "runs-on: [self-hosted,windows,x64,neral-gpu]" in text
    assert "github.ref == 'refs/heads/feat/qlora-v2-full-and-luna'" in text
