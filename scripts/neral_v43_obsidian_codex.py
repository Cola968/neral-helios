"""V4.3 Obsidian Codex: test-first offline Docker engine and bounded repair.

No candidate is executed on the host. This feature is DISABLED by default.
The pre-provisioned container image must contain pytest and Hypothesis.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import time

IMAGE = "neral-v43-codex-sandbox:1"
SAFE_IMPORTS = frozenset({
    "math", "itertools", "functools", "collections", "statistics", "heapq",
    "bisect", "decimal", "fractions", "re", "string", "operator",
})
FORBIDDEN = frozenset({"eval", "exec", "compile", "open", "__import__",
                      "input", "breakpoint", "globals", "locals", "vars",
                      "getattr", "setattr", "delattr"})
PROPERTY_KINDS = frozenset({"monotonic_int", "idempotent_int", "nonnegative_int"})
_CONTAINER_ENTRY = r'''
import contextlib, io, json, os, sys
from pathlib import Path
data = json.load(sys.stdin)
folder=Path("/tmp/neral-v43-codex")
folder.mkdir(mode=0o700, exist_ok=False)
(folder/"candidate.py").write_text(data["code"],encoding="utf-8")
(folder/"cases.json").write_text(json.dumps({"tests":data["tests"],
                                            "properties":data["properties"]}),
                               encoding="utf-8")
(folder/"test_candidate.py").write_text("""
import json, pytest
from pathlib import Path
from hypothesis import given, settings, strategies as st
import candidate

DATA=json.loads((Path(__file__).parent/"cases.json").read_text("utf-8"))

@pytest.mark.parametrize("case",DATA["tests"])
def test_examples(case):
    target=getattr(candidate,case["function"])
    actual=target(*case["args"])
    assert actual == case["expected"]

@pytest.mark.parametrize("spec",DATA["properties"])
@settings(max_examples=48,deadline=None,derandomize=True,database=None)
@given(data=st.data())
def test_properties(spec,data):
    x=data.draw(st.integers(min_value=spec["min"],max_value=spec["max"]))
    target=getattr(candidate,spec["function"])
    if spec["kind"]=="monotonic_int" and x<spec["max"]:
        assert target(x)<=target(x+1)
    elif spec["kind"]=="idempotent_int":
        assert target(target(x))==target(x)
    elif spec["kind"]=="nonnegative_int":
        assert target(x)>=0
""",encoding="utf-8")
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.path.insert(0,str(folder))
import pytest
output=io.StringIO()
with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
    rc=pytest.main(["-q","-x","--disable-warnings","-p","no:cacheprovider",
                    str(folder/"test_candidate.py")])
log=output.getvalue()
if rc==0:
    print("NERAL_V43_TESTS_PASS")
else:
    print("NERAL_V43_PROPERTY_COUNTEREXAMPLE" if "Falsifying example:" in log
          else "NERAL_V43_TESTS_FAILED")
    print(log[-2000:])
sys.exit(0 if rc==0 else 1)
'''


def validate_candidate(code: str) -> bool:
    if type(code) is not str or not 1 <= len(code.encode("utf-8")) <= 12000:
        raise ValueError("Invalid source size")
    try:
        tree=ast.parse(code,mode="exec")
    except SyntaxError as exc:
        raise ValueError("Python syntax_invalid") from exc
    if len(list(ast.walk(tree)))>4000:
        raise ValueError("AST too large")
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=(item.name for item in node.names) if isinstance(node,ast.Import) else (node.module,)
            if any(not m or m.split(".")[0] not in SAFE_IMPORTS for m in modules):
                raise ValueError("Disallowed module import")
            if isinstance(node,ast.ImportFrom) and node.level:
                raise ValueError("Relative import denied")
        if isinstance(node,ast.Name) and node.id in FORBIDDEN:
            raise ValueError("Disallowed host capability")
        if isinstance(node,ast.Attribute) and (
                node.attr.startswith("__") or node.attr in FORBIDDEN):
            raise ValueError("Dynamic attribute denied")
    return True


def _valid_case(case):
    if (type(case) is not dict or set(case)!={"function","args","expected"}
            or type(case["function"]) is not str
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}",case["function"])
            or type(case["args"]) is not list or len(case["args"])>8):
        raise ValueError("Invalid fixed testcase")


def _valid_property(rule):
    if (type(rule) is not dict or set(rule)!={"function","kind","min","max"}
            or type(rule["function"]) is not str
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}",rule["function"])
            or rule["kind"] not in PROPERTY_KINDS
            or type(rule["min"]) is not int or type(rule["max"]) is not int
            or not -1000<=rule["min"]<=rule["max"]<=1000):
        raise ValueError("Invalid property case")


def docker_command() -> list[str]:
    """No mounts, no network, unprivileged and no image pulling."""
    return ["docker","run","--rm","-i","--pull","never",
            "--network","none","--read-only","--memory","256m",
            "--memory-swap","256m","--cpus","1","--pids-limit","64",
            "--cap-drop","ALL","--security-opt","no-new-privileges",
            "--user","65534:65534",
            "--tmpfs","/tmp:rw,nosuid,nodev,size=48m",
            "--env","PYTHONDONTWRITEBYTECODE=1",
            IMAGE,"python","-B","-c",_CONTAINER_ENTRY]


def run_test_repair_loop(code, *, tests, properties, repair=None,
                         max_repairs=1, runner=subprocess.run, deadline=None):
    """Run tests before acceptance, repair from bounded sandbox stacktraces.

    'repair' must be a user-/model-provided callable; no arbitrary model or
    online API is invoked here. Any repaired code undergoes the same checks.
    """
    if (type(tests) is not list or type(properties) is not list
            or len(tests)>64 or len(properties)>16
            or type(max_repairs) is not int or not 0<=max_repairs<=1):
        raise ValueError("Invalid test or repair budget")
    for case in tests: _valid_case(case)
    for spec in properties: _valid_property(spec)
    # A no-test configuration cannot produce a verified output.
    if not tests and not properties:
        return {"status":"tests_failed","attempts":0,"reason":"no_tests","code":code,"verified":False}
    try:
        data=json.dumps({"code":code,"tests":tests,"properties":properties},
                        ensure_ascii=False,sort_keys=True,allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise ValueError("Unserializable test case") from exc
    if len(data.encode("utf-8"))>50000:
        raise ValueError("Oversized sandbox payload")
    original=code
    attempt=0
    end=min(time.monotonic()+60,deadline if deadline is not None else float("inf"))
    def result(status, *, accepted=False):
        return {"status":status,"attempts":attempt,
                "code":code if accepted else original,
                "verified":accepted}
    while attempt<=max_repairs:
        remaining=end-time.monotonic()
        if remaining<=0: return result("timeout")
        attempt+=1
        try:
            validate_candidate(code)
        except ValueError:
            return result("syntax_invalid")
        payload=json.dumps({"code":code,"tests":tests,"properties":properties},
                           ensure_ascii=False,sort_keys=True,allow_nan=False)
        try:
            process=runner(docker_command(),input=payload,text=True,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           check=False,timeout=min(30,remaining))
        except subprocess.TimeoutExpired:
            return result("timeout")
        except OSError:
            return result("sandbox_error")
        output=(process.stdout or "")[:4096]
        if process.returncode==0 and "NERAL_V43_TESTS_PASS" in output:
            return result("passed",accepted=True)
        if process.returncode in (125,126,127):
            return {"status":"sandbox_error","attempts":attempt}
        status=("property_counterexample"
                if "NERAL_V43_PROPERTY_COUNTEREXAMPLE" in output
                else "tests_failed" if "NERAL_V43_TESTS_FAILED" in output
                else "sandbox_error")
        if (status=="sandbox_error" or repair is None or attempt>max_repairs
                or not ("AssertionError" in output or "Falsifying example:" in output)):
            return result(status)
        if end-time.monotonic()<5:
            return result("timeout")
        if status=="sandbox_error":
            return result(status)
        # Stacktrace is UNTRUSTED data. Caller must not execute it. Limit bytes;
        # reparations are re-validated and run in the same Docker sandbox.
        safe_trace=output[-2000:]
        replacement=repair(code,safe_trace,attempt)
        if time.monotonic()>=end: return result("timeout")
        if type(replacement) is not str:
            return {"status":"syntax_invalid","attempts":attempt}
        code=replacement
    return result("tests_failed")
