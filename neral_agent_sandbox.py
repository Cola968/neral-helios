"""Run existing Python unit tests ONLY in an offline, read-only local Docker sandbox.

Never run LLM-generated code or subprocesses on the host. Docker image must
already be installed: docker --pull never. No network access from container.
Copy only explicitly allowed .py files into an ephemeral directory.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from agent_mode import AgentEngine

IMAGE = "python:3.11-alpine"
MAX_FILES = 100
MAX_COPY_BYTES = 4 * 1024 * 1024


def run_isolated_python_tests(engine: AgentEngine) -> dict:
    all_python = engine._files({".py"})
    tests = [name for name, _ in all_python if name.startswith("tests/test_") and name.endswith(".py")]
    if not tests:
        return {"status": "no_tests", "passed": None, "sandboxed": False,
                "note": "Keine freigegebenen tests/test_*.py gefunden."}
    try:
        image = subprocess.run(["docker", "image", "inspect", IMAGE],
                               capture_output=True, timeout=8, check=False)
        if image.returncode != 0:
            return {"status": "unavailable", "passed": None, "sandboxed": False,
                    "note": "Docker-Image nicht lokal installiert; keine Downloads."}
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "passed": None, "sandboxed": False,
                "note": "Lokaler Docker-Sandboxdienst nicht verfügbar."}

    with tempfile.TemporaryDirectory(prefix="neral-agent-tests-") as tmp:
        workspace = Path(tmp)
        total = 0
        copied = 0
        for name, source in all_python[:MAX_FILES]:
            try:
                raw = source.read_bytes()
            except OSError:
                continue
            if total + len(raw) > MAX_COPY_BYTES:
                break
            total += len(raw)
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            copied += 1
        try:
            workspace.chmod(0o755)
            for item in workspace.rglob("*.py"):
                item.chmod(0o644)
        except OSError:
            pass

        cmd = [
            "docker", "run", "--rm", "--pull", "never",
            "--network", "none", "--read-only",
            "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--mount", f"type=bind,src={workspace.resolve()},dst=/work,readonly",
            "--workdir", "/work",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            IMAGE, "python", "-B", "-m", "unittest", "discover",
            "-s", "tests", "-p", "test_*.py",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=45, check=False)
            if result.returncode in (125, 126, 127):
                return {"status": "unavailable", "passed": None, "sandboxed": False,
                        "note": "Docker-Container konnte nicht gestartet werden."}
            # Test output is untrusted. Keep a bounded excerpt, never execute or parse it as commands.
            out = (result.stdout + result.stderr).decode("utf-8", errors="replace")[-1500:]
            return {"status": "passed" if result.returncode == 0 else "failed",
                    "passed": result.returncode == 0, "sandboxed": True,
                    "files": copied, "output_tail": out}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "passed": False, "sandboxed": True,
                    "note": "Sandbox-Zeitbudget erreicht."}
        except OSError:
            return {"status": "unavailable", "passed": None, "sandboxed": False,
                    "note": "Docker konnte nicht ausgeführt werden."}
