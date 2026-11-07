"""Read-only MBPP functional scorer for NERAL-EXT saved responses.

Generated Python and benchmark assertions run ONLY inside an existing, offline,
read-only Docker image. No unsafe host execution, model inference or training.
Only aggregate status counts may be exported; private prompts/tests/responses
remain in the original Windows research directory.

This is an exploratory rerating of a previously consumed holdout; it is NOT
evidence of unseen-generalization or a new independent evaluation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess
import tempfile

from scripts.neral_ext import _mcnemar, read_cases, read_responses, write_report

IMAGE = "python:3.11-alpine"
VARIANTS = ("base", "base_fewshot", "neral_v5")
STATUS = ("passed", "failed", "timeout", "unavailable",
          "censored", "no_code", "invalid_python", "bad_contract")
FENCE = chr(96) * 3
FENCED = re.compile(
    r"(?s)" + re.escape(FENCE) + r"(?:python|py)?[ \t]*\r?\n"
    + r"(?P<code>.*?)\r?\n?" + re.escape(FENCE), re.IGNORECASE,
)


def prepare_candidate(raw: str) -> str | None:
    """Extract only one Python code block or a standalone function; no guessing."""
    if not isinstance(raw, str) or not 1 <= len(raw) <= 4096:
        return None
    if any((ord(ch) < 32 and ch not in "\r\n\t") or ord(ch) == 127
           for ch in raw):
        return None
    if FENCE in raw:
        if raw.count(FENCE) != 2:
            return None
        match = FENCED.search(raw)
        if match is None:
            return None
        raw = match.group("code")
    code = raw.strip()
    return code + "\n" if code and len(code) <= 4096 else None


def _validate_contract(case: dict, meta: dict, response: dict) -> None:
    if (not isinstance(case, dict) or case.get("kind") != "python"
            or not isinstance(meta, dict) or meta.get("id") != case.get("id")
            or meta.get("stratum") != "A"
            or not isinstance(meta.get("source"), str)
            or not meta["source"].startswith("mbpp:test:")
            or meta.get("sandbox_required") is not True):
        raise ValueError("Unexpected MBPP source identity")
    tests = meta.get("tests")
    setup = meta.get("test_setup_code")
    if (not isinstance(tests, list) or not 1 <= len(tests) <= 16
            or any(not isinstance(t, str) or not 7 <= len(t) <= 1000
                   or not t.lstrip().startswith("assert ") for t in tests)
            or not isinstance(setup, str) or len(setup) > 4000
            or not isinstance(response, dict)
            or set(response) != {"text", "complete"}
            or not isinstance(response["text"], str)
            or len(response["text"]) > 4096
            or type(response["complete"]) is not bool):
        raise ValueError("Invalid MBPP test or saved-response contract")


def _harness(tests: list[str], setup: str, marker: str) -> str:
    """Compose harness text; it will execute exclusively inside Docker."""
    # Literals are serialized using repr; arbitrary statements are never
    # interpolated as Python source in the host process.
    return (
        "import sys, traceback\n"
        + "statements = " + repr(tests) + "\n"
        + "setup = " + repr(setup) + "\n"
        + "scope = {'__name__':'__neral_mbpp_subject__'}\n"
        + "try:\n"
        + "    with open('/work/subject.py', encoding='utf-8') as stream:\n"
        + "        subject = stream.read()\n"
        + "    exec(compile(subject, '/work/subject.py', 'exec'),scope,scope)\n"
        + "    if setup:\n"
        + "        exec(compile(setup, '<mbpp_setup>', 'exec'),scope,scope)\n"
        + "    for statement in statements:\n"
        + "        exec(compile(statement, '<mbpp_assert>', 'exec'),scope,scope)\n"
        + "except BaseException:\n"
        + "    traceback.print_exc(limit=2)\n"
        + "    raise SystemExit(1)\n"
        + "sys.stdout.write(" + repr(marker + "\n") + ")\n"
    )


def grade_one(case: dict, meta: dict, response: dict) -> dict:
    """Run a single saved MBPP response; result exposes a scalar status only."""
    _validate_contract(case, meta, response)
    if not response["complete"]:
        return {"status": "censored"}
    code = prepare_candidate(response["text"])
    if code is None:
        return {"status": "no_code"}
    try:
        parsed = ast.parse(code, filename="<candidate>", mode="exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return {"status": "invalid_python" if re.search(r"\b(?:def|class|import|from)\b", code)
                else "no_code"}
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               for node in parsed.body):
        return {"status": "no_code"}
    try:
        image = subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True,
            timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable"}
    if image.returncode != 0:
        return {"status": "unavailable"}

    with tempfile.TemporaryDirectory(prefix="neral-mbpp-private-") as folder:
        work = Path(folder)
        marker = "MBPP_DONE_" + secrets.token_hex(16)
        (work / "subject.py").write_text(code, encoding="utf-8")
        (work / "runner.py").write_text(_harness(meta["tests"],
                                                   meta["test_setup_code"], marker),
                                        encoding="utf-8")
        try:
            work.chmod(0o755)
            for f in (work / "subject.py", work / "runner.py"):
                f.chmod(0o644)
        except OSError:
            pass
        cmd = [
            "docker", "run", "--rm", "--pull", "never",
            "--network", "none", "--read-only",
            "--memory", "128m", "--cpus", "0.5", "--pids-limit", "32",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=8m",
            "--mount", f"type=bind,src={work.resolve()},dst=/work,readonly",
            "--workdir", "/work", "--env", "PYTHONDONTWRITEBYTECODE=1",
            IMAGE, "python", "-I", "-B", "/work/runner.py",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=10, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        except OSError:
            return {"status": "unavailable"}
    if result.returncode in (125, 126, 127):
        return {"status": "unavailable"}
    success = (result.returncode == 0
               and result.stdout.splitlines().count(marker.encode("ascii")) == 1)
    return {"status": "passed" if success else "failed"}


def score_saved_mbpp(cases: list[dict], metadata: list[dict],
                     responses: dict[str, dict[str, dict]]) -> dict:
    """Aggregate results with strict identity. Never export candidate code."""
    if (not isinstance(cases, list) or not isinstance(metadata, list)
            or len(cases) != len(metadata) or len(cases) > 500
            or set(responses) != set(VARIANTS)):
        raise ValueError("Unmatched MBPP input")
    keyed = {}
    for record in metadata:
        if record.get("id") in keyed:
            raise ValueError("Duplicate metadata")
        keyed[record["id"]] = record
    if set(keyed) != {case["id"] for case in cases}:
        raise ValueError("Wrong metadata IDs")
    subset = [c for c in cases if c.get("kind") == "python"]
    if not subset:
        raise ValueError("No MBPP Python tasks in dataset")
    ids = {c["id"] for c in cases}
    summary = {}
    outcomes = {name: {} for name in VARIANTS}
    for variant in VARIANTS:
        data = responses[variant]
        if set(data) != ids:
            raise ValueError("Missing or unexpected variant response")
        counts = Counter()
        for case in subset:
            result = grade_one(case, keyed[case["id"]], data[case["id"]])
            outcomes[variant][case["id"]] = result["status"]
            counts[result["status"]] += 1
        out = {status: counts[status] for status in STATUS}
        out["total"] = len(subset)
        out["gradable_completed"] = out["passed"] + out["failed"]
        out["functional_attempted"] = (out["passed"] + out["failed"]
                                       + out["timeout"])
        out["scoring_complete"] = not any(out[s] for s in ("unavailable", "bad_contract"))
        out["timeout_count_explicit"] = out["timeout"]
        if sum(out[s] for s in STATUS) != len(subset):
            raise ValueError("Inconsistent MBPP status partition")
        summary[variant] = out
    both = [case["id"] for case in subset
            if all(outcomes[name][case["id"]] in ("passed", "failed")
                   for name in ("base_fewshot", "neral_v5"))]
    improvements = sum(outcomes["base_fewshot"][i] == "failed"
                       and outcomes["neral_v5"][i] == "passed" for i in both)
    regressions = sum(outcomes["base_fewshot"][i] == "passed"
                      and outcomes["neral_v5"][i] == "failed" for i in both)
    paired = {
        "both_tested": len(both),
        "excluded_ambiguous_or_censored": len(subset) - len(both),
        "improvements": improvements,
        "regressions": regressions,
        "exact_mcnemar_p_exploratory": (
            _mcnemar(improvements, regressions) if both else None),
        "significance_claim_allowed": False,
    }
    return {
        "schema": 1, "status": "NERAL_EXT_MBPP_FUNCTIONAL_DIAGNOSTIC",
        "paired_fewshot_vs_neral": paired,
        "significance_claim_allowed": False,
        "python_cases": len(subset), "by_variant": summary,
        "evaluated_saved_responses": True,
        "release_validated": False, "general_improvement_proven": False,
        "weight_change": False, "model_inference": False,
        "coding_solved_as_new_ability_proven": False,
        "caveat": "Existing consumed test set, incomplete responses censored; Docker tests are an exploratory functional check, not proof of security or unseen generalization.",
    }


def main(argv: list[str] | None = None) -> None:
    """Read pinned local 400-case files and write only an aggregate report."""
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("cases", "case-sha256", "manifest", "base",
                 "fewshot", "neral", "adapter-sha256", "out"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args(argv)
    if re.fullmatch(r"[a-f0-9]{64}", args.adapter_sha256) is None:
        p.error("Adapter identity must be pinned")
    cases = read_cases(Path(args.cases), args.case_sha256)
    source = Path(args.manifest)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 4_000_000:
        p.error("Untrusted local provenance manifest")
    meta = json.loads(source.read_text(encoding="utf-8"))
    if meta.get("dataset_sha256") != args.case_sha256:
        p.error("Mismatch between local dataset and manifest")
    result = score_saved_mbpp(
        cases, meta["cases"],
        {name: read_responses(Path(path)) for name, path in [
            ("base", args.base), ("base_fewshot", args.fewshot),
            ("neral_v5", args.neral)]})
    result["case_sha256"] = args.case_sha256
    result["adapter_sha256"] = args.adapter_sha256
    result["raw_response_hashes"] = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in [("base", args.base), ("base_fewshot", args.fewshot),
                           ("neral_v5", args.neral)]
    }
    write_report(Path(args.out), result)
    print("NERAL_EXT_MBPP_SCALAR_COMPLETE; raw source and tests never uploaded")


if __name__ == "__main__":
    main()
