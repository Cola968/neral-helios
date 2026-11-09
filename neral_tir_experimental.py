"""Explicit, OFF-BY-DEFAULT tool-integrated reasoning experiment.

Only a local Ollama model may propose one of two bounded tools. The caller must
enforce NERAL_EXPERIMENTAL_TIR=1, a per-request opt-in and the existing chat
lock. No generated Python runs on the Windows host. A clean Python exit only
proves execution; it does NOT prove the model translated the task correctly.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Callable

from neral_high_verified_first import split_task_and_format_instruction

# Qwen3.5's built-in Ollama parser treats <tool_call> as native XML and can
# return HTTP 500 / silently discard its JSON contents. Use an application-
# specific textual envelope that the native parser does not intercept.
_TOOL_CALL = re.compile(r"\s*NERAL_EXECUTE_JSON:\s*(?P<data>\{[\s\S]{1,4400}\})\s*\Z")
_INTEGER = re.compile(r"[+-]?\d{1,15}\Z")
_ALLOWED_TOOLS = frozenset({"solve", "python_sandbox"})
_FINAL_PREFIX = "NERAL_FINAL_TEXT:"
_PROMPT = (
    "\nEXPERIMENTAL LOCAL TIR: You have two safe, local tools. "
    "For a user question whose answer requires exact arithmetic, counting "
    "combinations or permutations, Boolean assignments, time-zone arithmetic, "
    "or checking Python code, YOU MUST propose a computation BEFORE guessing. "
    "Use solve ONLY for arithmetic expressions or linear equations; "
    "use python_sandbox for other computations or Python checks. "
    "There are NO native Ollama function tools in this request. "
    "Write exactly one plain-text line starting with NERAL_EXECUTE_JSON: "
    "followed by a strict JSON object, without Markdown or XML tags. "
    "Example for arithmetic: NERAL_EXECUTE_JSON: "
    '{\"tool\":\"solve\",\"code\":\"17*13\"} '
    "Example for other computations: NERAL_EXECUTE_JSON: "
    '{\"tool\":\"python_sandbox\",\"code\":\"print(17*13)\"} '
    "No punctuation after the final JSON brace. "
    "Use only tool and code keys; all code must be self-contained "
    "standard-library Python with a single printed result. "
    "Never invent execution results. For conversation or knowledge "
    "questions without necessary computation, emit NERAL_FINAL_TEXT: "
    "followed by the answer instead of a tool packet. Do not return "
    "unprefixed guesses for computational requests."
)
# The retry is task-agnostic. No benchmark wording or regex route.
_CHOICE_RETRY = (
    "Your previous answer did not contain an explicit tool decision. "
    "Reconsider the ORIGINAL user question. If exact computation, counting "
    "or running a program is needed, do not guess: emit exactly one "
    "NERAL_EXECUTE_JSON: object using solve for basic expressions or "
    "python_sandbox for other calculations. Otherwise emit "
    "NERAL_FINAL_TEXT: followed by the ordinary answer. "
    "Use no Markdown, preamble, native XML, or additional commentary."
)


def parse_tool_call(answer: str) -> tuple[str, str] | None:
    if not isinstance(answer, str) or len(answer.encode("utf-8")) > 5000:
        return None
    match = _TOOL_CALL.fullmatch(answer)
    if match is None:
        return None
    try:
        packet = json.loads(match.group("data"))
    except (TypeError, ValueError):
        return None
    if (type(packet) is not dict or set(packet) != {"tool", "code"}
            or packet["tool"] not in _ALLOWED_TOOLS
            or type(packet["code"]) is not str
            or not 1 <= len(packet["code"].encode("utf-8")) <= 4096):
        return None
    if any(ord(c) < 32 and c not in "\n\r\t" for c in packet["code"]):
        return None
    return packet["tool"], packet["code"]


def _run_ephemeral_python(code: str) -> dict:
    """Execute only inside preinstalled Docker: net=none, immutable FS, 5 sec.

    Local temporary material only contains the generated program and is never
    uploaded to GitHub. No host Python execution and no docker pull.
    """
    if not 1 <= len(code.encode("utf-8")) <= 4096:
        return {"status": "FAILED", "error_type": "RuntimeError",
                "traceback": "Input exceeds sandbox limit"}
    try:
        tree = ast.parse(code)
        if sum(1 for _ in ast.walk(tree)) > 384:
            raise ValueError("Too many AST nodes")
    except (SyntaxError, RecursionError, ValueError) as exc:
        return {"status": "FAILED", "error_type": "SyntaxError",
                "traceback": type(exc).__name__ + ": invalid program"}
    from neral_agent_sandbox import IMAGE
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "UNAVAILABLE", "error_type": "SandboxUnavailable"}
    if inspect.returncode != 0:
        return {"status": "UNAVAILABLE", "error_type": "SandboxUnavailable"}
    with tempfile.TemporaryDirectory(prefix="neral-tir-offline-") as folder:
        root = Path(folder)
        program = root / "run.py"
        program.write_text(code, encoding="utf-8")
        try:
            root.chmod(0o755)
            program.chmod(0o644)
        except OSError:
            return {"status": "UNAVAILABLE", "error_type": "SandboxUnavailable"}
        cmd = [
            "docker", "run", "--rm", "--pull", "never",
            "--network", "none", "--read-only",
            "--memory", "128m", "--memory-swap", "128m",
            "--cpus", "0.5", "--pids-limit", "32",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "--mount", f"type=bind,src={root.resolve()},dst=/work,readonly",
            "--workdir", "/work", "--env", "PYTHONDONTWRITEBYTECODE=1",
            IMAGE, "python", "-B", "run.py",
        ]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, check=False, timeout=5)
        except subprocess.TimeoutExpired:
            return {"status": "FAILED", "error_type": "RuntimeError",
                    "traceback": "Sandbox timeout (5s)"}
        except OSError:
            return {"status": "UNAVAILABLE", "error_type": "SandboxUnavailable"}
    if completed.returncode in (125, 126, 127):
        return {"status": "UNAVAILABLE", "error_type": "SandboxUnavailable"}
    if len(completed.stdout) > 4096 or len(completed.stderr) > 4096:
        return {"status": "FAILED", "error_type": "RuntimeError",
                "traceback": "Sandbox output exceeded limit"}
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", "replace")[-600:]
        return {"status": "FAILED", "error_type": "RuntimeError",
                "traceback": error or "Program exited with nonzero status"}
    result = completed.stdout.decode("utf-8", "replace").strip()
    if not result or len(result) > 512 or any(
            ord(c) < 32 and c not in "\t\n" for c in result):
        return {"status": "FAILED", "error_type": "RuntimeError",
                "traceback": "Empty or invalid program output"}
    return {"status": "PASSED", "output": result,
            "verification_scope": "execution_only"}


def execute_tir_tool(tool: str, code: str) -> dict:
    if tool == "python_sandbox":
        return _run_ephemeral_python(code)
    if tool == "solve":
        try:
            from solve import solve
            result = solve(code)
            if result.get("engine") not in (
                    "bounded-fraction-arithmetic", "bounded-linear-equation"):
                raise ValueError("Unrecognized local solver")
            value = result.get("result")
            if not isinstance(value, str) or not 1 <= len(value) <= 512:
                raise ValueError("No bounded solver result")
            return {"status": "PASSED", "output": value,
                    "verification_scope": "generated_expression_only"}
        except (ValueError, SyntaxError, OverflowError, RecursionError) as exc:
            return {"status": "FAILED", "error_type": "RuntimeError",
                    "traceback": type(exc).__name__ + ": invalid expression"}
    return {"status": "UNAVAILABLE", "error_type": "UnsupportedTool"}


def _display_output(output: str, question: str) -> str:
    core, suffix = split_task_and_format_instruction(question)
    value = output.strip()
    if suffix and _INTEGER.fullmatch(value):
        return "FINAL_ANSWER: " + value
    return value


def run_experimental_tir(
    messages: list[dict],
    respond: Callable[[list[dict]], str],
    *,
    tool_runner: Callable[[str, str], dict] = execute_tir_tool,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """At most two LLM responses and two sandbox attempts, never a tool chain.

    Python execution success is deliberately UNVERIFIED with respect to the
    question. Caller alone controls feature-gate, model identity and lock.
    """
    if (not isinstance(messages, list) or len(messages) < 2
            or messages[0].get("role") != "system"
            or messages[-1].get("role") != "user"
            or any(type(m.get("content")) is not str for m in messages)
            or not callable(respond) or not callable(tool_runner)):
        raise ValueError("Invalid experimental TIR input")
    started = clock()
    turns = [dict(row) for row in messages]
    # A benchmark's terminal output-format line is an output contract, not
    # part of the computational problem. Strip only the one exact, trusted
    # suffix; keep the original for output formatting after tool execution.
    core_task, format_suffix = split_task_and_format_instruction(messages[-1]["content"])
    if format_suffix:
        turns[-1]["content"] = core_task
    turns[0]["content"] += _PROMPT
    triggered = succeeded = 0
    tool_used = None
    last_error = "No verified tool result"
    unstructured_first = None

    def direct_unverified(value: str, calls: int) -> dict:
        return {
            "text": _display_output(value, messages[-1]["content"]),
            "mode": "HIGH", "model_generated": True,
            "verified": False, "verification_status": "not_available",
            "verification_scope": "none", "calls_used": calls,
            "tool_calls_triggered": 0, "tool_calls_succeeded": 0,
            "seconds": round(clock()-started, 3),
        }

    for attempt in range(2):
        answer = respond(turns)
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 32768:
            raise ValueError("Invalid local model response")
        parsed = parse_tool_call(answer)
        if parsed is None:
            if answer.startswith(_FINAL_PREFIX):
                content = answer[len(_FINAL_PREFIX):].strip()
                if (content and "NERAL_EXECUTE_JSON:" not in content
                        and "<tool_call" not in content and "</tool_call>" not in content):
                    return direct_unverified(content, attempt + 1)
                break
            # Never reinterpret malformed tool packets as valid direct answers.
            if ("NERAL_EXECUTE_JSON:" in answer or "<tool_call" in answer
                    or "</tool_call>" in answer):
                break
            if attempt == 0:
                # A single model-driven retry replaces premature acceptance
                # of a plain estimate. Never route by benchmark keywords.
                unstructured_first = answer
                turns.extend([
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": _CHOICE_RETRY},
                ])
                continue
            # After one retry, keep the first textual reply unverified rather
            # than treating an unstructured second reply as tool execution.
            if unstructured_first is not None:
                return direct_unverified(unstructured_first, 2)
            break
        tool, code = parsed
        tool_used = "neral_math_exact" if tool == "solve" else "neral_python_sandbox"
        triggered += 1
        try:
            result = tool_runner(tool, code)
        except Exception:
            # Sandbox errors must never expose local paths or private prompts.
            result = {"status": "UNAVAILABLE", "error_type": "SandboxUnavailable"}
        if not isinstance(result, dict):
            result = {"status": "UNAVAILABLE", "error_type": "SandboxUnavailable"}
        if result.get("status") == "PASSED" and isinstance(result.get("output"), str) and result["output"].strip():
            succeeded += 1
            return {
                "text": _display_output(result["output"], messages[-1]["content"]),
                "mode": "HIGH", "model_generated": False,
                "verified": False, "verification_status": "executed_unverified",
                "verification_scope": result.get("verification_scope", "execution_only"),
                "tool_used": tool_used, "calls_used": attempt+1,
                "tool_calls_triggered": triggered,
                "tool_calls_succeeded": succeeded,
                "seconds": round(clock()-started, 3),
            }
        if result.get("status") == "UNAVAILABLE":
            last_error = "Local sandbox unavailable"
            break
        last_error = str(result.get("traceback", "Tool error"))[-500:]
        if attempt == 0:
            turns.extend([
                {"role": "assistant", "content": answer},
                {"role": "user", "content": (
                    "Der lokale Toolaufruf schlug fehl: " + last_error +
                    "\nRepariere den Code genau einmal. Gib ausschließlich einen "
                    "neuen NERAL_EXECUTE_JSON:-Einzeiler zurück. Kein XML. Keine Shell."
                )},
            ])
    return {
        "text": "Die Werkzeugausführung konnte die Aufgabe nicht zuverlässig lösen.",
        "mode": "HIGH", "model_generated": False,
        "verified": False, "verification_status": "unverified_fallback",
        "verification_scope": "none",
        "tool_used": tool_used, "calls_used": min(2, 1+int(triggered > 1)),
        "tool_calls_triggered": triggered, "tool_calls_succeeded": succeeded,
        "seconds": round(clock()-started, 3),
    }
