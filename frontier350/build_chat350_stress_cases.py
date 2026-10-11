"""Deterministic, evaluation-only procedural generalization probes for Neral 350M.

Never add these generated prompts or answers to pretraining, SFT or DPO datasets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

SCHEMA_VERSION = 1


def _case(category: str, identifier: str, prompt: str, scoring: dict, limit: int = 48) -> dict:
    return {
        "id": identifier,
        "category": category,
        "prefix": f"User: {prompt}\nAssistant:",
        "max_new_tokens": limit,
        "scoring": scoring,
    }


def generate_cases(seed: int = 3502026) -> list[dict]:
    """Generate 56 distinct, deterministic evaluation cases."""
    rng = random.Random(seed)
    cases = []
    for i in range(16):
        a = rng.randint(18, 489)
        b = rng.randint(10, 290)
        op = i % 4
        if op == 0:
            prompt = f"Berechne {a} + {b}. Gib das Ergebnis als Zahl an."
            expected = a + b
        elif op == 1:
            a, b = max(a, b), min(a, b)
            prompt = f"Compute {a} minus {b}. Give only the numeric result."
            expected = a - b
        elif op == 2:
            a, b = rng.randint(3, 29), rng.randint(3, 19)
            prompt = f"Wie viel ist {a} mal {b}? Antworte mit der Zahl."
            expected = a * b
        else:
            a, b = rng.randint(2, 27), rng.randint(3, 19)
            prompt = f"What is {a} times {b}? Reply with the result."
            expected = a * b
        cases.append(_case("math", f"stress-math-{i:02d}", prompt,
                           {"type": "integer", "value": expected}, 32))

    tokens = ["BERLIN", "KOMET", "SATURN", "NORD", "ORBIT", "SIGNAL",
              "VECTOR", "NEBEL", "RADAR", "TAU", "MOND", "PHYSIK"]
    rng.shuffle(tokens)
    for i, token in enumerate(tokens):
        prompt = (f"Antworte exakt mit {token} und füge nichts hinzu." if i % 2 == 0 else
                  f"Reply with exactly {token}. Do not add any other text.")
        cases.append(_case("instruction_exact", f"stress-exact-{i:02d}", prompt,
                           {"type": "exact", "value": token}, 14))

    words = ["Smaragd", "Opal", "Kobalt", "Granit", "Atlas", "Quarz",
             "Nebel", "Kranich", "Vega", "Lumen"]
    rng.shuffle(words)
    for i in range(12):
        number = rng.randint(2100, 9899)
        word = words[i // 2]
        if i % 2 == 0:
            prefix = (f"User: Bitte merke dir für dieses Gespräch das Kennwort {word}.\n"
                      f"Assistant: Verstanden.\nUser: Wie lautete das Kennwort?\nAssistant:")
            needle = word
        else:
            prefix = (f"User: For this conversation the number is {number}.\n"
                      f"Assistant: Okay.\nUser: What number did I just give you?\nAssistant:")
            needle = str(number)
        cases.append({"id": f"stress-memory-{i:02d}", "category": "context_memory",
                      "prefix": prefix, "max_new_tokens": 28,
                      "scoring": {"type": "contains_all", "values": [needle]}})

    requirements = [
        ("def sum_two", "return", "+"),
        ("def double_value", "return", "*"),
        ("def square_value", "return", "*"),
        ("def subtract_two", "return", "-"),
        ("def triple_value", "return", "*"),
        ("def absolute_value", "return", "abs("),
        ("def cube_value", "return", "**"),
        ("def add_five", "return", "+"),
    ]
    instructions = [
        "Return a Python function sum_two(a, b) that adds its inputs.",
        "Schreibe eine Python-Funktion double_value(n), die n verdoppelt.",
        "Implement a Python function square_value(n) that squares n.",
        "Schreibe die Funktion subtract_two(a, b), die b von a subtrahiert.",
        "Write Python function triple_value(n) returning three times n.",
        "Schreibe eine Python-Funktion absolute_value(n) mit abs().",
        "Write a Python function cube_value(n) that computes n cubed.",
        "Schreibe die Python-Funktion add_five(n), die 5 addiert.",
    ]
    for i, (req, instruction) in enumerate(zip(requirements, instructions)):
        cases.append(_case("code", f"stress-code-{i:02d}", instruction,
                           {"type": "contains_all", "values": list(req)}, 95))

    translations = [
        ("Übersetze ins Englische: Die Sonne ist ein Stern.", ["sun", "star"]),
        ("Translate to German: The train is late.", ["zug"], ["spät", "verspät"]),
        ("Übersetze ins Englische: Der Himmel ist blau.", ["sky", "blue"]),
        ("Translate to German: The planet is small.", ["planet", "klein"]),
        ("Übersetze ins Englische: Das Flugzeug startet heute.", ["today"],
         ["aircraft", "airplane", "plane"]),
        ("Translate to German: The water is cold.", ["wasser", "kalt"]),
        ("Übersetze ins Englische: Der Mond umkreist die Erde.", ["moon", "earth"]),
        ("Translate to German: The test is finished.", ["test"],
         ["fertig", "abgeschlossen", "beendet"]),
    ]
    for i, entry in enumerate(translations):
        prompt, needles = entry[0], entry[1]
        scoring = ({"type": "contains_all", "values": needles} if len(entry) == 2 else
                   {"type": "contains_any", "values": entry[2], "and_values": needles})
        cases.append(_case("translation", f"stress-translation-{i:02d}", prompt, scoring, 60))

    if len(cases) != 56:
        raise AssertionError(f"unexpected generated test count: {len(cases)}")
    if len({r["id"] for r in cases}) != len(cases):
        raise AssertionError("test IDs must be unique")
    if len({r["prefix"] for r in cases}) != len(cases):
        raise AssertionError("generated test questions must be distinct")
    return cases


def write_cases(seed: int, out: Path) -> dict:
    cases = generate_cases(seed)
    if out.exists():
        raise FileExistsError("evaluation output must be fresh")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cases, indent=2, ensure_ascii=False) + "\n"
    out.write_text(payload, encoding="utf8")
    digest = hashlib.sha256(payload.encode("utf8")).hexdigest()
    return {"schema": SCHEMA_VERSION, "seed": seed, "cases": len(cases),
            "sha256": digest, "policy": "evaluation-only; never use as training data"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3502026)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(write_cases(args.seed, Path(args.out))))


if __name__ == "__main__":
    main()
