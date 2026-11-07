"""NERAL v5: new, CPU-verifiable transfer-curriculum candidate.

Six quantitative families (75% of training) and two coding families. Supports
concise answers and, *only when asked*, checkable final-answer reasoning.
No pretrained checkpoint, benchmark answer, GPU, network, optimizer or inference.
NOT a human-reviewed dataset, independent external benchmark, or training grant.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

SEED = 20261108
PER_STYLE = 40
CATEGORIES = (
    "affine", "fractions", "money", "fuel", "timezone", "quantities",
    "python_trace", "python_debug",
)
STYLES = (
    "Löse genau: {task}\nAntworte ausschließlich als {fmt}.",
    "Rechne sorgfältig: {task}\nAusgabe nur als {fmt}; keine Erklärung.",
    "Technischer Auftrag: {task}\nNur {fmt} zurückgeben.",
    "Bestimme das Ergebnis: {task}\nGib {fmt}, sonst nichts.",
    "Prüfe rechnerisch: {task}\nLiefere CHECK und danach FINAL_ANSWER.",
    "Rechne unabhängig nach: {task}\nSchreibe CHECK und FINAL_ANSWER.",
    "Unabhängige Testformulierung: {task}\nErgebnis ausschließlich als {fmt}.",
    "Fachaufgabe: {task}\nNur {fmt} ohne Zusatztext.",
)
ROWS = {"train": 1920, "val": 320, "test": 320}
TOTAL = len(CATEGORIES) * len(STYLES) * PER_STYLE
SYSTEM = (
    "Du bist NERAL. Löse technische Aufgaben korrekt. "
    "Halte dich an das jeweils verlangte Antwortformat. "
    "Wenn die Aufgabe CHECK und FINAL_ANSWER verlangt, prüfe die Rechnung."
)
ANNOTATED = (1, 4, 5, 7)
WORKED = (4, 5)


def _rng(category: str, style: int, index: int, seed: int) -> random.Random:
    digest = hashlib.sha256(
        f"NERAL-v5-transfer|{seed}|{category}|{style}|{index}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _validate(category: str, style: int, index: int, seed: int) -> None:
    if (category not in CATEGORIES or type(style) is not int or
            not 0 <= style < len(STYLES) or type(index) is not int or
            not 0 <= index < PER_STYLE or type(seed) is not int or
            not 0 <= seed < 2**63):
        raise ValueError("Invalid generator coordinates")


def _round_half_up_nonnegative(value: Fraction) -> int:
    if value < 0:
        raise ValueError("Only nonnegative money can be rounded")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def solve_case(category: str, style: int, index: int, seed: int = SEED) -> dict:
    """Construct a fresh problem and oracle without using evaluation cases."""
    _validate(category, style, index, seed)
    rng = _rng(category, style, index, seed)
    annotated = style in ANNOTATED
    result = {}

    if category == "affine":
        a, b = index + 4, rng.randint(2, 13)
        left_shift, right_shift = rng.randint(-18, 20), rng.randint(-12, 14)
        x = rng.randint(-65, 65)
        constant = a * left_shift + b * right_shift
        target = (a + b) * x + constant
        assert a * (x + left_shift) + b * (x + right_shift) == target
        result.update(
            task=(f"Finde x: {a}(x + ({left_shift})) + "
                  f"{b}(x + ({right_shift})) = {target}."),
            answer=f"x = {x}" if annotated else str(x),
            fmt="x = Zahl" if annotated else "Zahl ohne x oder Einheit",
            check=f"{a}*({x}+({left_shift}))+{b}*({x}+({right_shift}))={target}",
        )

    elif category == "fractions":
        numerator = index - 19
        denominator = rng.randint(2, 23)
        other_numerator = rng.randint(-29, 31)
        other_denominator = rng.randint(2, 27)
        operator = ("+", "-", "*", "/")[index % 4]
        if operator == "/" and other_numerator == 0:
            other_numerator = 7
        left = Fraction(numerator, denominator)
        right = Fraction(other_numerator, other_denominator)
        if operator == "+":
            value = left + right
        elif operator == "-":
            value = left - right
        elif operator == "*":
            value = left * right
        else:
            value = left / right
        result.update(
            task=(f"Berechne den gekürzten Bruch ({numerator}/{denominator}) "
                  f"{operator} ({other_numerator}/{other_denominator})."),
            answer=str(value),
            fmt="gekürzter Bruch oder ganze Zahl",
            check=f"{left} {operator} {right} = {value}",
            left=str(left), right=str(right), operator=operator,
        )

    elif category == "money":
        source_cents = 185 + index * 139 + style * 17 + rng.randint(0, 9)
        discount = rng.randint(3, 54)
        surcharge = rng.randint(2, 38)
        exact_cents = Fraction(
            source_cents * (100 - discount) * (100 + surcharge), 10_000
        )
        final_cents = _round_half_up_nonnegative(exact_cents)
        price = f"{source_cents // 100},{source_cents % 100:02d}"
        amount = f"{final_cents // 100},{final_cents % 100:02d}"
        result.update(
            task=(f"Ein Artikel kostet {price} EUR. Erst {discount}% Rabatt, "
                  f"danach {surcharge}% Aufschlag auf den verminderten Preis. "
                  "Berechne den Endpreis kaufmännisch auf Cent gerundet."),
            answer=amount + (" EUR" if annotated else ""),
            fmt="Betrag mit zwei Dezimalstellen und EUR" if annotated
                else "Betrag mit zwei Dezimalstellen, ohne Währung",
            check=f"Exakte Cent vor Rundung: {exact_cents}; gerundet: {final_cents}",
            source_cents=source_cents, discount=discount, surcharge=surcharge,
        )

    elif category == "fuel":
        burn = rng.randint(18, 62) * 30
        duration = (index + 3) * 15
        reserve = rng.randint(2, 11) * 10
        taxi = rng.randint(9, 145)
        value = Fraction(burn * (duration + reserve), 60) + taxi
        result.update(
            task=(f"Treibstoffplanung: Verbrauch {burn} kg/h; Flug "
                  f"{duration} Minuten; Reserve {reserve} Minuten; "
                  f"Rollen {taxi} kg. Berechne die gesamte Treibstoffmasse exakt."),
            answer=str(value) + (" kg" if annotated else ""),
            fmt="exakter Kilogrammwert mit kg" if annotated
                else "exakter Zahlenwert ohne Einheit",
            check=f"{burn}*({duration}+{reserve})/60+{taxi} = {value}",
            burn=burn, duration=duration, reserve=reserve, taxi=taxi,
        )

    elif category == "timezone":
        hour, minute = rng.randint(0, 23), 15 * rng.randint(0, 3)
        src, dst = rng.randint(-8, 9), rng.randint(-7, 11)
        duration = (index + 2) * 15
        minutes = 60 * hour + minute + duration + 60 * (dst - src)
        day, clock = divmod(minutes, 1440)
        bare = f"{clock // 60:02d}:{clock % 60:02d} Tag {day:+d}"
        answer = bare.replace(" Tag ", " (Tag ") + ")" if annotated else bare
        result.update(
            task=(f"Feste UTC-Offsets ohne Sommerzeit: Start {hour:02d}:{minute:02d} "
                  f"bei UTC{src:+d}; Flug {duration} Minuten; Ziel UTC{dst:+d}. "
                  "Bestimme Ankunftszeit und Tagesverschiebung."),
            answer=answer,
            fmt="HH:MM (Tag +/-N)" if annotated else "HH:MM Tag +/-N",
            check=f"Minutenzähler {minutes}; Tag {day:+d}; Tagesminute {clock}",
            hour=hour, minute=minute, src=src, dst=dst, duration=duration,
        )

    elif category == "quantities":
        rate = index + 8
        hours = rng.randint(2, 14)
        rejected = rng.randint(1, 18)
        bonus = rng.randint(1, 15)
        value = rate * hours - rejected + bonus
        result.update(
            task=(f"Eine Maschine produziert {rate} Teile pro Stunde über "
                  f"{hours} Stunden. {rejected} Teile fallen bei der Prüfung "
                  f"aus, danach werden {bonus} geprüfte Teile ergänzt. "
                  "Wie viele verwendbare Teile sind vorhanden?"),
            answer=str(value) + (" Stück" if annotated else ""),
            fmt="ganze Stückzahl mit Stück" if annotated
                else "ganze Zahl ohne Einheit",
            check=f"{rate}*{hours}-{rejected}+{bonus}={value}",
        )

    elif category == "python_trace":
        n = index + 3
        kind = index % 8
        if kind == 0:
            program = f"def f(v):\n    return sum(range(v))\nprint(f({n}))"
            value = sum(range(n))
        elif kind == 1:
            upper = n + 5
            program = (f"def f(v):\n    return min(max(v, 4), {upper})\n"
                       f"print(f({n * 2}))")
            value = min(max(n * 2, 4), upper)
        elif kind == 2:
            values = [n % 11, (n + 7) % 19, (n + 15) % 23]
            program = (f"def f(values):\n    return max(values) - min(values)\n"
                       f"print(f({values!r}))")
            value = max(values) - min(values)
        elif kind == 3:
            div = rng.randint(2, 12)
            program = (f"def f(v):\n    return int(v % {div} == 0)\n"
                       f"print(f({n}))")
            value = int(n % div == 0)
        elif kind == 4:
            program = f"def f(v):\n    return len(str(v))\nprint(f({n * n}))"
            value = len(str(n * n))
        elif kind == 5:
            div = rng.randint(2, 11)
            program = (f"def f(v):\n    return v // {div} + v % {div}\n"
                       f"print(f({n}))")
            value = n // div + n % div
        elif kind == 6:
            program = (f"def f(v):\n    return (v + 2) * (v - 2) - v * v\n"
                       f"print(f({n}))")
            value = -4
        else:
            program = f"def f(v):\n    return sum(range(1, v, 2))\nprint(f({n}))"
            value = sum(range(1, n, 2))
        result.update(
            task=("Welcher einzelne Integer wird bei diesem Python-Programm "
                  "gedruckt? Kein Programm ausführen.\n" + program),
            answer=str(value),
            fmt="ein einzelner Integer",
            program=program,
        )

    else:
        name = f"repair_{style}_{index}"
        kind = index % 8
        if kind == 0:
            buggy = f"def {name}(items):\n    return min(items) if items else 0"
            fixed = f"def {name}(items):\n    return max(items) if items else 0"
            goal = "größten Eintrag zurückgeben, sonst 0"
        elif kind == 1:
            buggy = f"def {name}(limit):\n    return sum(range(limit + 1))"
            fixed = f"def {name}(limit):\n    return sum(range(limit))"
            goal = "Summe aller Integer vor limit zurückgeben"
        elif kind == 2:
            buggy = f"def {name}(value, low, high):\n    return min(low, max(value, high))"
            fixed = f"def {name}(value, low, high):\n    return max(low, min(value, high))"
            goal = "value auf das Intervall [low, high] begrenzen"
        elif kind == 3:
            buggy = f"def {name}(items):\n    return items[1] if items else None"
            fixed = f"def {name}(items):\n    return items[0] if items else None"
            goal = "erstes Element oder None bei leerer Liste"
        elif kind == 4:
            buggy = f"def {name}(value, threshold):\n    return value > threshold"
            fixed = f"def {name}(value, threshold):\n    return value >= threshold"
            goal = "True ab Erreichen des Grenzwerts"
        elif kind == 5:
            buggy = f"def {name}(text):\n    return text == text[::1]"
            fixed = f"def {name}(text):\n    return text == text[::-1]"
            goal = "ein Palindrom am umgekehrten Text erkennen"
        elif kind == 6:
            offset = rng.randint(1, 16)
            buggy = f"def {name}(v):\n    return v * 3 - {offset}"
            fixed = f"def {name}(v):\n    return v * 3 + {offset}"
            goal = f"drei mal v plus {offset} berechnen"
        else:
            div = rng.randint(2, 23)
            buggy = f"def {name}(v):\n    return v % {div} == 1"
            fixed = f"def {name}(v):\n    return v % {div} == 0"
            goal = f"Teilbarkeit durch {div} korrekt erkennen"
        tree = ast.parse(fixed)
        if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
                or len(list(ast.walk(tree))) > 75):
            raise ValueError("Unexpected reviewed Python repair AST")
        result.update(
            task=f"Repariere den Fehler. Ziel: {goal}.\n{buggy}",
            answer=fixed,
            fmt="ausschließlich die korrigierte Python-Funktionsdefinition",
            program=fixed,
        )
    return result


def make_case(category: str, style: int, index: int, seed: int = SEED) -> dict:
    sol = solve_case(category, style, index, seed)
    if category in CATEGORIES[:6] and style in WORKED:
        completion = f"CHECK: {sol['check']}\nFINAL_ANSWER: {sol['answer']}"
    else:
        completion = sol["answer"]
    if category in CATEGORIES[-2:] and style in WORKED:
        prompt = (f"Prüfe und löse die Code-Aufgabe {style}:{index}:\n"
                  f"{sol['task']}\nGib ausschließlich {sol['fmt']} zurück.")
    else:
        prompt = STYLES[style].format(task=sol["task"], fmt=sol["fmt"])
    if (not completion.strip() or len(completion) > 400
            or len(prompt) > 600):
        raise ValueError("Example exceeds reviewed candidate budget")
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "completion": [{"role": "assistant", "content": completion}],
    }


def _generated(seed: int) -> dict[str, bytes]:
    payloads: dict[str, list[str]] = {key: [] for key in ROWS}
    seen: set[str] = set()
    for category in CATEGORIES:
        for style in range(len(STYLES)):
            split = "train" if style < 6 else ("val" if style == 6 else "test")
            for index in range(PER_STYLE):
                row = make_case(category, style, index, seed)
                signature = " ".join(row["prompt"][1]["content"].casefold().split())
                if signature in seen:
                    raise ValueError("Duplicate prompt across v5 splits")
                seen.add(signature)
                payloads[split].append(
                    json.dumps(row, ensure_ascii=False,
                               sort_keys=True, separators=(",", ":")) + "\n"
                )
    if len(seen) != TOTAL:
        raise ValueError("Incomplete new training candidate")
    for split, expected in ROWS.items():
        if len(payloads[split]) != expected:
            raise ValueError("Unbalanced or leaking training split")
    return {name: "".join(rows).encode("utf8")
            for name, rows in payloads.items()}


def _manifest(seed: int, generated: dict[str, bytes]) -> dict:
    return {
        "schema_version": 5,
        "dataset": "NERAL v5 quantitative transfer candidate; not yet trained",
        "seed": seed,
        "categories": list(CATEGORIES),
        "rows": dict(ROWS),
        "sha256": {split: hashlib.sha256(data).hexdigest()
                   for split, data in generated.items()},
        "split_policy": "Style-group-disjoint: six train, one validation, one test",
        "evidence": "Exact Fraction and integer arithmetic plus bounded static Python AST",
        "source": "new deterministic v5 generator; no v4 benchmark examples or labels",
        "train_math_ratio": "6/8",
        "training_allowed": False,
        "human_quality_reviewed": False,
        "tokenizer_verified": False,
        "training_performed": False,
        "quality_improvement_proven": False,
        "independent_external_holdout": False,
        "limitations": [
            "All examples are synthetic templates; transfer is unproven.",
            "Validation and test share problem families and are not external benchmarks.",
            "Explicitly request checks to receive multi-line reasoning.",
            "Never use holdout cases as adapter training inputs.",
        ],
    }


def build(root: Path, seed: int = SEED) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Invalid v5 generation seed")
    root = Path(root)
    if root.is_symlink() or (root.exists() and (
            not root.is_dir() or any(root.iterdir()))):
        raise ValueError("Output must be an empty nonlinked directory")
    outputs = _generated(seed)
    root.mkdir(parents=True, exist_ok=True)
    for split, data in outputs.items():
        (root / (split + ".jsonl")).write_bytes(data)
    manifest = _manifest(seed, outputs)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    return manifest


def verify(root: Path) -> dict:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing nonlinked v5 candidate directory")
    required = {"train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"}
    if ({p.name for p in root.iterdir()} != required
            or any(p.is_symlink() or not p.is_file() for p in root.iterdir())):
        raise ValueError("Missing or unexpected candidate source")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    seed = manifest.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Unapproved deterministic seed")
    expected = _generated(seed)
    if manifest != _manifest(seed, expected):
        raise ValueError("Manifest does not match trusted v5 generator")
    for split, data in expected.items():
        path = root / (split + ".jsonl")
        if path.stat().st_size > 3_000_000 or path.read_bytes() != data:
            raise ValueError("Candidate labels or prompts differ from verified source")
    return {"schema": 5, "rows": dict(ROWS), "valid": True,
            "answers_verified": True, "training_allowed": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", default=SEED, type=int)
    a = p.parse_args(argv)
    result = build(a.out, seed=a.seed)
    assert verify(a.out)["valid"]
    print(json.dumps({"dataset": result["dataset"], "rows": result["rows"],
                      "sha256": result["sha256"], "training_allowed": False},
                     sort_keys=True))


if __name__ == "__main__":
    main()
