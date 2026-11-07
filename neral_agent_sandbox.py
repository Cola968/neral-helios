"""Run existing Python unit tests ONLY in an offline, read-only local Docker sandbox.

Never run LLM-generated code or subprocesses on the host. Docker image must
already be installed: docker --pull never. No network access from container.
Copy only explicitly allowed .py files into an ephemeral directory.
"""
from __future__ import annotations

import secrets
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


def _validated_assertions(cases):
    """Only simple literal comparisons to direct function calls are accepted.

    The syntax is parsed on host; no supplied Python expression is evaluated
    outside the networkless Docker container.
    """
    import ast
    if not isinstance(cases, list) or not 1 <= len(cases) <= 16:
        raise ValueError("1–16 erlaubte assert-Fälle erforderlich.")
    allowed_literals=(ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
                      ast.UnaryOp, ast.USub, ast.UAdd)
    lines=[]
    for source in cases:
        if not isinstance(source,str) or not 7<=len(source)<=300 or not source.startswith("assert "):
            raise ValueError("Ungültiger Testfall.")
        try:tree=ast.parse(source,mode="exec")
        except (SyntaxError,RecursionError) as exc:raise ValueError("Ungültiger Python-Assert.") from exc
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.Assert):
            raise ValueError("Nur ein einzelner Assert zulässig.")
        cond=tree.body[0].test
        if not (isinstance(cond,ast.Compare) and len(cond.ops)==1
                and isinstance(cond.ops[0],(ast.Eq,ast.Is))
                and len(cond.comparators)==1 and isinstance(cond.left,ast.Call)):
            raise ValueError("Nur direkter Funktionsaufruf == Wert bzw. is None.")
        fn=cond.left
        if not isinstance(fn.func,ast.Name) or fn.func.id.startswith("_") or fn.keywords:
            raise ValueError("Nur öffentliche Funktionen ohne Keywords.")
        try:
            for arg in fn.args:ast.literal_eval(arg)
            ast.literal_eval(cond.comparators[0])
        except (ValueError,TypeError,RecursionError,MemoryError) as exc:
            raise ValueError("Nur begrenzte literale Testargumente.") from exc
        if len(list(ast.walk(tree)))>72:raise ValueError("Testfall zu komplex.")
        lines.append(source)
    return lines


def run_isolated_unit_test(engine: AgentEngine, file_path: str, cases: list[str],
                           *, timeout: int = 3) -> dict:
    """Return test outcome and bounded exact assertion traceback from local Docker.

    No fallback to host Python. Failure, Docker-unavailable or timeout are NOT
    success; no source/program/traceback content leaves the local PC via CI.
    """
    import ast
    if timeout != 3:
        raise ValueError("Testzeitlimit fest auf 3 Sekunden begrenzt.")
    tests=_validated_assertions(cases)
    target=engine.safe_path(file_path)
    if target.suffix!=".py" or not target.is_file():
        raise ValueError("Nur existierende Projekt-Python-Dateien.")
    raw=target.read_bytes()
    if len(raw)>32_768:raise ValueError("Projektdatei zu groß.")
    try:ast.parse(raw,filename=file_path)
    except SyntaxError as exc:
        return {"status":"SYNTAX_FAILED","passed":False,"sandboxed":False,
                "error_type":"SyntaxError","line":exc.lineno}
    cmd_check=["docker","image","inspect",IMAGE]
    try:
        check=subprocess.run(cmd_check,capture_output=True,timeout=3,check=False)
    except (OSError,subprocess.TimeoutExpired):
        return {"status":"UNAVAILABLE","passed":False,"sandboxed":False,
                "note":"Lokales Offline-Docker-Image fehlt."}
    if check.returncode!=0:
        return {"status":"UNAVAILABLE","passed":False,"sandboxed":False,
                "note":"Das zuvor installierte Offline-Docker-Image fehlt."}
    with tempfile.TemporaryDirectory(prefix="neral-unit-sandbox-") as folder:
        work=Path(folder)
        (work/"subject.py").write_bytes(raw)
        completion_marker = "NERAL_ASSERTIONS_COMPLETED:" + secrets.token_hex(16)
        # An exit code of zero can also mean the subject exited before tests.
        # This marker detects accidental early exits, not a malicious subject
        # deliberately forging the runner's output from inside the container.
        harness=("try:\n    from subject import *\n"+
                 ''.join("    "+test+"\n" for test in tests)+
                 "except BaseException:\n    import traceback\n    traceback.print_exc()\n"
                 "    raise SystemExit(1)\n"
                 "print()\n"+f"print({completion_marker!r}, flush=True)\n").encode("utf-8")
        (work/"runner.py").write_bytes(harness)
        for name in ("subject.py","runner.py"):
            try:(work/name).chmod(0o644)
            except OSError:pass
        try:work.chmod(0o755)
        except OSError:pass
        cmd=["docker","run","--rm","--pull","never","--network","none",
             "--read-only","--memory","128m","--cpus","0.5","--pids-limit","32",
             "--cap-drop","ALL","--security-opt","no-new-privileges",
             "--user","65534:65534","--tmpfs","/tmp:rw,noexec,nosuid,size=8m",
             "--mount",f"type=bind,src={work.resolve()},dst=/work,readonly",
             "--workdir","/work","--env","PYTHONDONTWRITEBYTECODE=1",
             IMAGE,"python","-B","runner.py"]
        try:
            proc=subprocess.run(cmd,capture_output=True,timeout=3,check=False)
        except subprocess.TimeoutExpired:
            return {"status":"TIMEOUT","passed":False,"sandboxed":True,"timeout_seconds":3}
        except OSError:
            return {"status":"UNAVAILABLE","passed":False,"sandboxed":False}
    if proc.returncode in (125,126,127):
        return {"status":"UNAVAILABLE","passed":False,"sandboxed":False}
    # Truncate untrusted error output. The agent can use this for local repair,
    # never for automatic execution on the Windows host.
    trace=(proc.stderr+proc.stdout).decode("utf-8",errors="replace")[-1800:]
    completed = proc.stdout.splitlines().count(completion_marker.encode("ascii")) == 1
    passed = proc.returncode == 0 and completed
    return {"status":"PASSED" if passed else "FAILED",
            "passed":passed,"sandboxed":True,"assertions_completed":completed,
            "assertions":len(tests),"traceback":trace,
            "source_sha256":engine._digest(raw)}
