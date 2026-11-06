"""NERAL v3 combined development corpus, verified and generated only locally.

744 inherited reviewed math-v2 rows already contain the 86 reviewed NERAL
core rows. Add 48 new exact affine equations and 50 synthetic tool decisions.
No training, downloads, model weights or evaluation answer uploads here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from neral_agent_math import solve_math_symbolic, equation_final_answer
from scripts.neral_math_v2_builder import norm, load_holdout_prompts
from scripts.neral_math_v2_eval import cases as math_cases
from scripts.neral_v2_generalization_probe import checked_cases
from scripts.neral_math_v3_balanced_pack import _read_checked, EXPECTED_V2_DATA

SEED = 9686112026
V2_ROWS, CORE_ALREADY_IN_V2 = 744, 86
EQUATIONS, TRAJECTORIES = 48, 50
TOTAL = V2_ROWS + EQUATIONS + TRAJECTORIES
SOURCE_KEY = "scripts/neral_v3_combined.py"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def line(system: str, user: str, assistant: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":"))


def equation_rows() -> list[str]:
    rng = random.Random(SEED)
    result = []
    seen = set()
    while len(result) < EQUATIONS:
        a = rng.choice([i for i in range(-18, 19) if abs(i) >= 3])
        c = rng.choice([i for i in range(-7, 8) if i and i != a])
        x = rng.randint(-70, 85)
        b = rng.randint(-135, 165)
        d = (a - c) * x + b
        equation = f"{a}x + ({b}) = {c}x + ({d})"
        if equation in seen:
            continue
        seen.add(equation)
        calc = solve_math_symbolic(equation)
        steps = "\n".join(calc["steps"])
        if calc["status"] != "PASSED" or calc["value"] != str(x):
            raise ValueError("New algebra example fails exact solution")
        if equation_final_answer(steps) != str(x):
            raise ValueError("New algebra example fails final answer contract")
        instruction = (
            f"Löse {equation}. Zeige höchstens zwei STEP-Zeilen und schließe "
            "mit FINAL_ANSWER: Zahl ab." if len(result) % 2 == 0 else
            f"Solve {equation}. Use at most two STEP lines, ending with FINAL_ANSWER: number."
        )
        result.append(line(
            "Du bist NERAL. Antworte mit kompakten STEP- und FINAL_ANSWER-Zeilen.",
            instruction, steps,
        ))
    return result


def trajectory_rows() -> list[str]:
    """Synthetic action decisions, not recordings or claims of real execution."""
    actions = [
        "inventory", "read", "search", "python", "propose_edit",
        "syntax_check", "run_isolated_unit_test", "finish",
        "solve_math_symbolic", "eval_python_expr",
    ]
    result = []
    for i in range(TRAJECTORIES):
        case_id = 500 + i
        action = actions[i % len(actions)]
        filename = f"scratch_{case_id}.py"
        conditions = {
            "inventory": "Es gibt noch kein Dateiinventar.",
            "read": f"Du musst die bekannte Datei {filename} lesen.",
            "search": f"Suche den Ausdruck demo_key_{case_id} im lokalen Projekt.",
            "python": "Die relevanten Dateien müssen statisch geprüft werden.",
            "propose_edit": f"Die Datei {filename} wurde gelesen und der SHA-Abgleich ist gültig; schlage einen Patch vor, keine automatische Speicherung.",
            "syntax_check": f"Der Patch auf {filename} wurde vom Nutzer freigegeben; AST-Syntaxprüfung ausstehend.",
            "run_isolated_unit_test": f"Der Syntaxcheck für {filename} ist PASSED; feste Nutzerassertions müssen im Offline-Docker-Container geprüft werden.",
            "finish": f"Syntax- und Docker-Prüfungen für Revision {case_id} waren PASSED. Antworte kurz und ehrlich.",
            "solve_math_symbolic": f"Prüfe die Gleichung 4x + {case_id} = {case_id + 24}.",
            "eval_python_expr": f"Rechne exakt: 1/{case_id} + 1/{case_id}.",
        }
        output = {"action": action}
        if action in {"read", "propose_edit"}:
            output["path"] = filename
        if action == "propose_edit":
            output["content"] = f"def demo_key_{case_id}(n):\n    return n + {case_id}\n"
        if action == "search":
            output["query"] = f"demo_key_{case_id}"
        if action == "solve_math_symbolic":
            output["equation"] = f"4x + {case_id} = {case_id + 24}"
        if action == "eval_python_expr":
            output["expression"] = f"1/{case_id} + 1/{case_id}"
        if action == "finish":
            output["summary"] = "Checks passed in this synthetic example; no production deployment."
        prompt = f"Synthetisches Agentenbeispiel {case_id}. Zustand: {conditions[action]} Wähle genau eine Aktion."
        result.append(line(
            "Du bist NERAL, ein lokaler Agent. Antworte mit genau einem JSON-Objekt. Kein Zugriff ohne Berechtigung.",
            prompt, json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        ))
    return result


def check_overlap(rows: list[str], holdout_path: Path, coding_suite_path: Path) -> dict:
    protected = {norm(x["prompt"]) for x in math_cases()}
    protected |= {norm(x[2]) for x in checked_cases()}
    protected |= set(load_holdout_prompts(holdout_path))
    coding = json.loads(coding_suite_path.read_text(encoding="utf-8"))
    if len(coding) != 12 or len({x["id"] for x in coding}) != 12:
        raise ValueError("Coding benchmark catalog differs from fixed 12-case contract")
    coding_spans = set()
    for item in coding:
        protected.add(norm(item["goal"]))
        coding_spans.add(norm(item["buggy_source"]))
        coding_spans.add(norm(item["goal"]))
        coding_spans.update(norm(case) for case in item["test_cases"])
    seen = set()
    for n, text in enumerate(rows, 1):
        rec = json.loads(text)
        chat = rec["messages"]
        if [m["role"] for m in chat] != ["system", "user", "assistant"]:
            raise ValueError(f"Invalid chat structure: {n}")
        prompt = norm(chat[1]["content"])
        if prompt in protected or prompt in seen:
            raise ValueError(f"Evaluation leakage or training duplicate: row {n}")
        seen.add(prompt)
        generated = norm(chat[2]["content"])
        if any(len(s) > 32 and (s in prompt or s in generated) for s in coding_spans):
            raise ValueError(f"Coding suite contamination: row {n}")
    return {
        "checked_training_rows": len(rows),
        "protected_eval_prompts": len(protected),
        "coding_tasks": len(coding),
        "exact_overlap": 0,
        "duplicate_prompts": 0,
        "verbatim_coding_overlap": 0,
        "semantic_leakage_excluded": False,
    }


def prepare(data_path: Path, review_path: Path, holdout_path: Path,
            coding_suite_path: Path, source_path: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("Never overwrite an existing local data directory")
    previous, review = _read_checked(data_path, review_path)
    if len(previous) != V2_ROWS or len(review["rows"]) != V2_ROWS:
        raise ValueError("The 744 reviewed input rows are required")
    additions = equation_rows() + trajectory_rows()
    combined = previous + additions
    if len(combined) != TOTAL:
        raise ValueError("Invalid combined corpus size")
    leakage = check_overlap(combined, holdout_path, coding_suite_path)
    sources = dict(review["sources"])
    if SOURCE_KEY in sources or source_path.is_symlink() or not source_path.is_file():
        raise ValueError("The new generator must be a newly checked source")
    sources[SOURCE_KEY] = digest(source_path.read_bytes())
    proof = list(review["rows"])
    for n, text in enumerate(additions, V2_ROWS + 1):
        proof.append({
            "line": n,
            "row_sha256": digest(text.encode("utf-8")),
            "evidence": [
                "exact_affine_solver_and_FINAL_ANSWER" if n <= V2_ROWS + EQUATIONS
                else "deterministic_synthetic_tool_state",
                "public_dev_not_holdout",
            ],
            "finding": "Deterministic CPU-contract generation; not independently human reviewed.",
        })
    data = ("\n".join(combined) + "\n").encode("utf-8")
    manifest = {
        "schema": 1,
        "dataset_sha256": digest(data),
        "sources": sources,
        "rows": proof,
        "reviewer": "Inherited reviewed v2 plus deterministic extensions; no model quality approval.",
    }
    review_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if len(data) > 1024 * 1024 or len(review_bytes) > 1024 * 1024:
        raise ValueError("Original trainer imposes a 1 MiB bound")
    output.mkdir(exist_ok=False)
    (output / "train.jsonl").write_bytes(data)
    (output / "train.review.json").write_bytes(review_bytes)
    summary = {
        "schema": 1,
        "status": "DATA_CANDIDATE_NOT_TRAINED",
        "examples": TOTAL,
        "inherited_math_v2": V2_ROWS,
        "reviewed_core_inherited_not_counted_twice": CORE_ALREADY_IN_V2,
        "new_equations": EQUATIONS,
        "new_synthetic_trajectories": TRAJECTORIES,
        "dataset_sha256": digest(data),
        "review_sha256": digest(review_bytes),
        "source_sha256": sources[SOURCE_KEY],
        "leakage": leakage,
        "trained": False,
        "release_ready": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("v2-data", "v2-review", "holdout", "coding-suite", "source", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    x = p.parse_args()
    print(json.dumps(prepare(
        x.v2_data, x.v2_review, x.holdout, x.coding_suite, x.source, x.out
    ), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
