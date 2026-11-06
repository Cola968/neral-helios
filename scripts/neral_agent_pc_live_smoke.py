"""Read-only REAL local-model agent smoke on Windows self-hosted RTX runner.

Never upload model data, prompts, generated text or local user files.
Only an allowlisted scalar summary is written to a GitHub Actions workspace file.
"""
from __future__ import annotations
import argparse
import ast
import json
import tempfile
import urllib.request
from pathlib import Path
from neral_agent_autonomy import AutonomousAgent, OllamaLocalBackend


def _ollama_available(name: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            raw = response.read(32769)
        if len(raw) > 32768:
            return False
        packet = json.loads(raw)
        return any(row.get("name") == name for row in packet.get("models", []))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def choose_backend(base: str, adapter: str, name: str):
    if base and adapter and Path(base).is_dir() and (Path(adapter) / "adapter_model.safetensors").is_file():
        from neral_agent_hf import HFAdapterLocalBackend
        return HFAdapterLocalBackend(base, adapter, max_new_tokens=320), "hf_adapter"
    if _ollama_available(name):
        return OllamaLocalBackend(name, timeout=65), "ollama"
    return None, "unavailable"


def smoke(backend):
    with tempfile.TemporaryDirectory(prefix="neral-agent-live-") as directory:
        root = Path(directory)
        (root / "calculation.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (root / "README.md").write_text("# Isolierter NERAL-Agententest\n", encoding="utf-8")
        agent = AutonomousAgent(root, backend, max_steps=4, max_seconds=480)
        started = agent.start(
            "Live-Test: Lies calculation.py mit dem read-Werkzeug, "
            "prüfe danach mit dem python-Werkzeug die Syntax und schließe "
            "mit finish ab. Keine Dateiänderungen.")
        result = agent.run(started["id"])
        names = [item["action"] for item in result["history"]]
        return {
            "status": result["status"],
            "steps": result["steps"],
            "read_performed": "read" in names,
            "python_check_performed": "python" in names,
            "finished": result["status"] == "completed",
            "writes_performed": 0,
            "result_correct": result["status"] == "completed" and "read" in names and "python" in names,
        }


def _exact_multiplication(source: str) -> tuple[bool, bool]:
    # Static AST shape, no generated Python is executed on Windows.
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False, False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return True, False
    fn = tree.body[0]
    if fn.name != "multiply" or [arg.arg for arg in fn.args.args] != ["a", "b"]:
        return True, False
    if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        return True, False
    expr = fn.body[0].value
    valid = isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult)
    valid = valid and isinstance(expr.left, ast.Name) and expr.left.id == "a"
    valid = valid and isinstance(expr.right, ast.Name) and expr.right.id == "b"
    return True, bool(valid)


def patch_smoke(backend):
    # The harness may approve a patch only inside this disposable temp directory.
    # Never auto-approve a patch in the real user's project.
    metrics = {
        "patch_proposed": False, "patch_approved_test_only": False,
        "patch_ast_valid": False, "patch_logic_verified": False,
        "patch_steps": 0, "executed_generated_code_on_host": False
    }
    with tempfile.TemporaryDirectory(prefix="neral-agent-patch-eval-") as directory:
        root = Path(directory)
        path = root / "buggy.py"
        path.write_bytes(b"def multiply(a, b):\n    return a + b\n")
        agent = AutonomousAgent(root, backend, max_steps=6, max_seconds=480)
        started = agent.start(
            "Bearbeite ein isoliertes Python-Testprojekt: Lies zuerst buggy.py. "
            "Die Funktion multiply(a,b) benutzt faelschlich Addition. "
            "Schlage per propose_edit den vollstaendigen korrigierten Inhalt "
            "von buggy.py vor, der a*b berechnet. Beende dann mit finish. "
            "Keine anderen Dateien bearbeiten.")
        result = agent.run(started["id"])
        metrics["patch_steps"] = result["steps"]
        proposal = result.get("proposal")
        if result["status"] == "awaiting_approval" and proposal and proposal.get("path") == "buggy.py":
            metrics["patch_proposed"] = True
            try:
                agent.approve(started["id"], proposal["token"])
                metrics["patch_approved_test_only"] = True
                agent.run(started["id"])
            except Exception:
                pass
        content = path.read_text(encoding="utf-8")
        metrics["patch_ast_valid"], metrics["patch_logic_verified"] = _exact_multiplication(content)
        return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="")
    p.add_argument("--adapter", default="")
    p.add_argument("--ollama-model", default="qwen3.5:9b")
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if a.report.exists():
        p.error("Refusing to overwrite existing report")
    report = {
        "schema": 1, "suite": "real_model_local_agent_smoke",
        "model_backend": "unavailable", "model_loaded": False,
        "adapter_loaded": False, "read_performed": False,
        "python_check_performed": False, "finished": False, "result_correct": False,
        "steps": 0, "writes_performed": 0, "no_cloud_model_calls": True,
        "training_performed": False, "code_execution_performed": False,
        "error_category": "none", "status": "not_started",
        "patch_proposed": False, "patch_approved_test_only": False,
        "patch_ast_valid": False, "patch_logic_verified": False,
        "patch_steps": 0, "executed_generated_code_on_host": False,
    }
    try:
        backend, provider = choose_backend(a.base, a.adapter, a.ollama_model)
        report["model_backend"] = provider
        if backend is None:
            report["error_category"] = "local_model_unavailable"
        else:
            observed = smoke(backend)
            report.update(observed)
            report["model_loaded"] = observed["steps"] > 0 and observed["status"] != "failed"
            report["adapter_loaded"] = provider == "hf_adapter" and getattr(backend, "_model", None) is not None
            if report["result_correct"]:
                # Bonus coding check, not a launch gate. Must never run
                # generated Python on the Windows host.
                try:
                    report.update(patch_smoke(backend))
                except Exception:
                    report["patch_steps"] = 0
            else:
                report["error_category"] = "model_tool_call_or_workflow_not_completed"
    except Exception:
        report["error_category"] = "local_inference_or_setup_failed"
        report["status"] = "failed"
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2), encoding="ascii")
    print("NERAL_LIVE_SCALAR_RESULT=" + json.dumps(report, sort_keys=True))
    return 0 if report["result_correct"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
