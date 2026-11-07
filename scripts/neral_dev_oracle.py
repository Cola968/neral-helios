"""Independent CPU oracles for NERAL development suites.

Code golds are checked by replaying the fenced program in a short-lived
CPython process after an AST allowlist. Other golds are recomputed from the
prompt text, not from the generator's index formula.

This module is for repository fixtures only. It is not a sandbox for model
output and it does not replace neral_agent_python_trace.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile

_CACHE: dict[str, int] = {}
_FENCE = re.compile(r"```python\n(.*?)```", re.S)
_INT = re.compile(r"[+-]?\d+\Z")
_ALLOWED_NODES = {
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If,
    ast.Continue, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.IfExp, ast.Call, ast.Name, ast.Constant, ast.Load, ast.Store,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.Compare, ast.Eq,
    ast.Lt, ast.Subscript, ast.Slice, ast.List, ast.Dict, ast.Attribute,
    ast.comprehension, ast.SetComp, ast.GeneratorExp, ast.Tuple,
}
_CALLS = {"print", "sum", "range", "len", "enumerate"}
_ATTRS = {"append", "values", "count"}
_BANNED_NAMES = {
    "exec", "eval", "open", "compile", "__import__", "globals", "locals",
    "vars", "input", "getattr", "setattr", "delattr", "type", "object",
    "super", "breakpoint", "help", "dir", "memoryview", "bytearray", "bytes",
    "import", "os", "sys", "subprocess", "builtins",
}


class OracleMismatch(ValueError):
    """Gold integer does not match an independent CPU oracle."""


def _reject(reason: str) -> None:
    raise ValueError("Fixture rejected before execution: " + reason)


def _range_ok(call: ast.Call) -> None:
    bounds = []
    for arg in call.args:
        if type(arg) is not ast.Constant or type(arg.value) is not int:
            _reject("range bounds must be integer constants")
        bounds.append(arg.value)
    if not 1 <= len(bounds) <= 3:
        _reject("range arity")
    if len(bounds) == 3 and bounds[2] == 0:
        _reject("range step")
    if len(range(*bounds)) > 5000:
        _reject("range longer than 5000")


def _validate(source: str) -> None:
    if (not isinstance(source, str) or not 1 <= len(source) <= 2000
            or "\x00" in source or "__" in source):
        _reject("empty, huge or private source")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        _reject("syntax")
    defined = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(defined) != sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)):
        _reject("nested function")
    prints = 0
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            _reject(type(node).__name__)
        if isinstance(node, ast.Name) and (
                node.id in _BANNED_NAMES or node.id.startswith("_")):
            _reject("name " + node.id)
        if isinstance(node, ast.Constant):
            if type(node.value) is bool or type(node.value) not in (int, str):
                _reject("constant")
            if type(node.value) is str and (len(node.value) > 200 or "__" in node.value):
                _reject("string")
            if type(node.value) is int and abs(node.value) > 1_000_000:
                _reject("integer")
        if isinstance(node, ast.Attribute) and node.attr not in _ATTRS:
            _reject("attribute")
        if isinstance(node, ast.Call):
            if node.keywords:
                _reject("keyword call")
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _CALLS and func.id not in defined:
                    _reject("call " + func.id)
                if func.id == "print":
                    prints += 1
                    if len(node.args) != 1:
                        _reject("print arity")
                if func.id == "range":
                    _range_ok(node)
                if func.id in {"sum", "len", "enumerate"} and len(node.args) != 1:
                    _reject(func.id + " arity")
            elif isinstance(func, ast.Attribute):
                receiver_ok = isinstance(func.value, ast.Name) or (
                    func.attr == "count" and isinstance(func.value, ast.Subscript))
                if func.attr not in _ATTRS or not receiver_ok:
                    _reject("method")
                if func.attr == "append" and len(node.args) != 1:
                    _reject("append arity")
                if func.attr == "values" and node.args:
                    _reject("values arity")
                if func.attr == "count" and len(node.args) != 1:
                    _reject("count arity")
            else:
                _reject("call shape")
        if isinstance(node, ast.AugAssign) and type(node.op) is not ast.Add:
            _reject("augmented operator")
        if isinstance(node, ast.BinOp) and type(node.op) not in (
                ast.Add, ast.Sub, ast.Mult, ast.Mod):
            _reject("operator")
        if isinstance(node, ast.Compare) and (
                len(node.ops) != 1 or len(node.comparators) != 1
                or type(node.ops[0]) not in (ast.Eq, ast.Lt)):
            _reject("comparison")
        if isinstance(node, ast.For) and node.orelse:
            _reject("for else")
        if isinstance(node, ast.FunctionDef) and (
                node.decorator_list or node.args.defaults or node.args.kwonlyargs
                or node.args.vararg or node.args.kwarg or node.args.posonlyargs
                or len(node.args.args) > 2):
            _reject("function")
    if prints != 1:
        _reject("expected exactly one print")


def _child_limits() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def replay_source(source: str) -> int:
    """Run one allowlisted fixture. Raises before any process if syntax is unsafe."""
    _validate(source)
    cached = _CACHE.get(source)
    if cached is not None:
        return cached
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONSAFEPATH": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C",
    }
    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT"):
            if os.environ.get(key):
                env[key] = os.environ[key]
    with tempfile.TemporaryDirectory(prefix="neral-oracle-") as folder:
        path = os.path.join(folder, "snippet.py")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(source if source.endswith("\n") else source + "\n")
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _child_limits
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-B", path],
                cwd=folder,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
                **kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            raise OracleMismatch("Fixture exceeded 2 seconds") from exc
    if done.returncode != 0 or done.stderr.strip():
        raise OracleMismatch("Fixture process failed")
    line = done.stdout.strip()
    if _INT.fullmatch(line) is None:
        raise OracleMismatch("Fixture did not print one integer")
    value = int(line)
    if abs(value) >= 1_000_000:
        raise OracleMismatch("Fixture integer out of range")
    _CACHE[source] = value
    return value


def fenced_source(prompt: str) -> str:
    if not isinstance(prompt, str):
        _reject("prompt")
    found = _FENCE.findall(prompt)
    if len(found) != 1:
        _reject("expected one python fence")
    return found[0]


def _nums(pattern: str, text: str) -> list[int]:
    match = re.search(pattern, text)
    if match is None:
        raise OracleMismatch("Prompt no longer matches the independent pattern")
    return [int(group) for group in match.groups()]


def _euclid(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _pow(base: int, exp: int, mod: int) -> int:
    if mod <= 1 or exp < 0:
        raise OracleMismatch("Invalid modular exponent")
    result, factor = 1, base % mod
    while exp:
        if exp & 1:
            result = result * factor % mod
        factor = factor * factor % mod
        exp >>= 1
    return result


def _coins(total: int) -> int:
    ways = 0
    for nines in range(total // 9 + 1):
        left = total - 9 * nines
        for fives in range(left // 5 + 1):
            if (left - 5 * fives) % 2 == 0:
                ways += 1
    return ways


def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        raise OracleMismatch("Invalid binomial")
    k = min(k, n - k)
    value = 1
    for i in range(k):
        value = value * (n - i) // (i + 1)
    return value


def _horn(a_known: int, b_known: int) -> int:
    known_a, known_b, known_c = bool(a_known), bool(b_known), False
    changed = True
    while changed:
        changed = False
        if known_a and not known_b:
            known_b = True
            changed = True
        if known_b and not known_c:
            known_c = True
            changed = True
    return int(known_c)


def _reach(edges: list[tuple[int, int]]) -> int:
    seen = {0}
    changed = True
    while changed:
        changed = False
        for src, dst in edges:
            if src in seen and dst not in seen:
                seen.add(dst)
                changed = True
    return len(seen) - 1


def prompt_oracle(case: dict) -> int:
    """Second procedure. Code uses process replay; other tasks parse the prompt."""
    if not isinstance(case, dict):
        raise OracleMismatch("Case")
    family = case.get("family")
    text = case.get("prompt")
    if not isinstance(text, str) or not isinstance(family, str):
        raise OracleMismatch("Case fields")
    if case.get("domain") == "code_trace" or "```python" in text:
        return replay_source(fenced_source(text))
    if family == "euclidean_gcd":
        a, b = _nums(r"(?:von|of) (\d+) (?:und|and) (\d+)", text)
        return _euclid(a, b)
    if family == "modular_exponent":
        base, exp, mod = _nums(
            r"(?:von|of) (\d+) (?:hoch|to the power) (\d+) "
            r"(?:geteilt durch|modulo) (\d+)", text)
        return _pow(base, exp, mod)
    if family == "coin_combinations":
        total, = _nums(r"2a\+5b\+9c=(\d+)", text)
        return _coins(total)
    if family == "square_digit_sum":
        value, = _nums(r"(?:Quadriere|Square) (\d+)", text)
        return sum(int(ch) for ch in str(value * value))
    if family == "four_switch_parity":
        raw = re.search(r"\[((?:0|1)(?:, (?:0|1))*)\]", text)
        if raw is None:
            raise OracleMismatch("Missing switch bits")
        bits = [int(part) for part in raw.group(1).split(", ")]
        bit = 0
        for item in bits:
            bit ^= item
        return bit
    if family == "directed_reachability":
        raw = re.search(r"\[(?:\(\d+, \d+\)(?:, )?)+\]", text)
        if raw is None:
            raise OracleMismatch("Missing edges")
        edges = ast.literal_eval(raw.group(0))
        return _reach([(int(a), int(b)) for a, b in edges])
    if family == "boolean_assignments":
        target, cap = _nums(
            r"\(A XOR B\)=(\d+).*?(?:höchstens|at most) (\d+)", text)
        count = 0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    if ((a ^ b) == target and (b or c) and not (a and c)
                            and a + b + c <= cap):
                        count += 1
        return count
    if family == "directed_shortest_path":
        if "Kantenkosten" in text:
            costs = _nums(
                r"Kantenkosten (\d+) und (\d+); über B,C mit (\d+), (\d+), (\d+); direkt mit (\d+)",
                text)
        else:
            costs = _nums(
                r"costing (\d+) and (\d+); via B,C costing (\d+), (\d+), (\d+); and direct costing (\d+)",
                text)
        return min(costs[0] + costs[1], costs[2] + costs[3] + costs[4], costs[5])
    if family == "choose_without_order":
        if text.startswith("Aus "):
            people, leaders = _nums(
                r"Aus (\d+) verschiedenen Personen werden genau (\d+)", text)
        else:
            leaders, people = _nums(r"Choose (\d+) people from (\d+) distinct", text)
        return _binom(people, leaders)
    if family == "modular_stepping":
        if "beginnt bei" in text:
            start, times, step, mod = _nums(
                r"beginnt bei (\d+)\. Er wird (\d+)-mal um (\d+) erhöht und anschließend durch (\d+)",
                text)
        else:
            start, step, times, mod = _nums(
                r"starts at (\d+), increases by (\d+) exactly (\d+) times, then is divided by (\d+)",
                text)
        return (start + times * step) % mod
    if family == "interval_divisibility":
        div, low, high = _nums(
            r"(?:durch|by) (\d+).*(?:zwischen|from) (\d+) (?:und|through) (\d+)",
            text)
        if div <= 0 or high < low:
            raise OracleMismatch("Invalid interval")
        return high // div - (low - 1) // div
    if family == "integer_equation_counts":
        total, = _nums(r"3a \+ 5b = (\d+)", text)
        return sum((total - 5 * b) % 3 == 0 for b in range(total // 5 + 1))
    if family == "boolean_three_var":
        a, b, c = _nums(r"A=(\d+), B=(\d+), C=(\d+)", text)
        return int((a and not b) or c)
    if family == "transitive_order":
        match = re.search(r"Steht (\w+) vor (\w+)", text) or re.search(
            r"Is (\w+) before (\w+)", text)
        if match is None:
            raise OracleMismatch("Missing order question")
        names = ("Ada", "Bea", "Cem", "Dora")
        left, right = match.group(1), match.group(2)
        if left not in names or right not in names:
            raise OracleMismatch("Unknown person")
        return int(names.index(left) < names.index(right))
    if family == "horn_entailment":
        a, b = _nums(r"A=(\d+), B=(\d+)", text)
        return _horn(a, b)
    if family == "set_outside_union":
        if "Grundmenge" in text:
            universe, in_a, in_b, both = _nums(
                r"Grundmenge (\d+) Elemente: A hat (\d+), B hat (\d+), die Schnittmenge hat (\d+)",
                text)
        else:
            universe, in_a, in_b, both = _nums(
                r"has (\d+) elements; set A has (\d+), B has (\d+), and their intersection has (\d+)",
                text)
        if not (0 <= both <= in_a and both <= in_b and in_a + in_b - both <= universe):
            raise OracleMismatch("Impossible set sizes")
        return universe - (in_a + in_b - both)
    raise OracleMismatch("No independent oracle for family " + family)


def assert_independent_oracles(cases: list[dict]) -> None:
    if not isinstance(cases, list) or not cases:
        raise OracleMismatch("Empty suite")
    for case in cases:
        expected = prompt_oracle(case)
        gold = case.get("answer")
        if not isinstance(gold, str) or _INT.fullmatch(gold) is None or int(gold) != expected:
            raise OracleMismatch(
                "Gold disagrees with independent oracle for " + str(case.get("id")))
