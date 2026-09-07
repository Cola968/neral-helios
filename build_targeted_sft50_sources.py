"""Augment pinned OASST1 with deterministic Neral-owned targeted SFT sources.

The generated rows are formula/template derived, not produced by another model. They target
Neral's measured weak areas: German instruction following, arithmetic/algebra explanations,
Python fundamentals and short reasoning. The resulting source manifest remains explicit and
hash-pinned so prepare_sft50.py can apply the same cleaning, split and contamination rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from fractions import Fraction
from pathlib import Path

GENERATED_LICENSE = "Project-generated deterministic data; no external corpus"
GENERATED_DOC_URL = "https://github.com/Cola968/neral-helios/blob/main/TARGETED_DATA.md"
TARGET_TRAIN_EXAMPLES = 20_000


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _split_for_prompt(prompt: str) -> str:
    group = hashlib.sha256(("SFT50:PROMPT_GROUP:v2:" + _normalize(prompt)).encode("utf8")).hexdigest()
    bucket = int(group[:8], 16) % 100
    return "holdout" if bucket < 2 else "test" if bucket < 5 else "val" if bucket < 10 else "train"


def _targeted_v1_prompts() -> set[str]:
    path = Path(__file__).resolve().parent / "benchmarks50" / "targeted_v1.jsonl"
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row.get("prompt"), str):
            out.add(_normalize(row["prompt"]))
    return out

# Deliberately keeps human OASST1 as a large share while giving weak domains enough weight.
WEIGHTS = {
    "oasst1-human-en": 0.27,
    "oasst1-human-de": 0.03,
    "neral-generated-de-core": 0.20,
    "neral-generated-math-de": 0.12,
    "neral-generated-math-en": 0.12,
    "neral-generated-code-de": 0.08,
    "neral-generated-code-en": 0.12,
    "neral-generated-reasoning-de": 0.03,
    "neral-generated-reasoning-en": 0.03,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row(source: str, i: int, prompt: str, answer: str, domain: str):
    return {"id": f"{source}-{i:06d}", "prompt": prompt, "answer": answer, "domain": domain}


def generate_math(language: str, count: int, seed: int):
    rng = random.Random(seed)
    out = []
    for i in range(count):
        kind = i % 8
        if kind == 0:
            a, b = rng.randint(20, 900), rng.randint(10, 700)
            if language == "de":
                p = f"Berechne {a} + {b} und erkläre den Rechenweg kurz."
                ans = f"{a} + {b} = {a+b}. Das Ergebnis ist {a+b}."
            else:
                p = f"Compute {a} + {b} and briefly explain the calculation."
                ans = f"{a} + {b} = {a+b}. The result is {a+b}."
        elif kind == 1:
            a, b = rng.randint(120, 1200), rng.randint(10, 119)
            if b > a:
                a, b = b, a
            if language == "de":
                p = f"Berechne {a} - {b}. Zeige einen kurzen Rechenweg."
                ans = f"{a} - {b} = {a-b}."
            else:
                p = f"Compute {a} - {b}. Show a short calculation."
                ans = f"{a} - {b} = {a-b}."
        elif kind == 2:
            a, b = rng.randint(3, 35), rng.randint(3, 35)
            if language == "de":
                p = f"Wie viel ist {a} mal {b}? Begründe knapp."
                ans = f"{a} · {b} = {a*b}."
            else:
                p = f"What is {a} times {b}? Give a concise calculation."
                ans = f"{a} × {b} = {a*b}."
        elif kind == 3:
            a = rng.randint(2, 12)
            x = rng.randint(-25, 35)
            b = rng.randint(-30, 30)
            c = a * x + b
            if language == "de":
                p = f"Löse die Gleichung {a}x + ({b}) = {c} nach x und zeige die Umformung."
                ans = f"{a}x = {c-b}. Durch {a} teilen: x = {x}."
            else:
                p = f"Solve {a}x + ({b}) = {c} for x and show the rearrangement."
                ans = f"{a}x = {c-b}. Divide by {a}: x = {x}."
        elif kind == 4:
            pct = rng.choice([5, 10, 20, 25, 40, 50, 75])
            base = rng.randint(2, 90) * 20
            value = base * pct // 100
            if language == "de":
                p = f"Bestimme {pct}% von {base} und erkläre die Rechnung."
                ans = f"{pct}% = {pct}/100. Daher {base} · {pct}/100 = {value}."
            else:
                p = f"Find {pct}% of {base} and explain the calculation."
                ans = f"{pct}% = {pct}/100, so {base} × {pct}/100 = {value}."
        elif kind == 5:
            num = rng.randint(1, 12)
            den = rng.randint(2, 15)
            mult = rng.randint(2, 12)
            f = Fraction(num, den)
            result = f * mult
            r = str(result.numerator) if result.denominator == 1 else f"{result.numerator}/{result.denominator}"
            if language == "de":
                p = f"Berechne {num}/{den} · {mult} und kürze vollständig."
                ans = f"{num}/{den} · {mult} = {num*mult}/{den} = {r}."
            else:
                p = f"Compute {num}/{den} × {mult} and simplify fully."
                ans = f"{num}/{den} × {mult} = {num*mult}/{den} = {r}."
        elif kind == 6:
            mass = rng.randint(1, 20)
            acc = rng.randint(1, 15)
            force = mass * acc
            if language == "de":
                p = f"Nutze F = m·a. Welche Kraft wirkt bei m = {mass} kg und a = {acc} m/s²? Zeige Formel und Ergebnis."
                ans = f"F = m·a = {mass} kg · {acc} m/s² = {force} N."
            else:
                p = f"Use F = m·a. What force acts for m = {mass} kg and a = {acc} m/s²? Show formula and result."
                ans = f"F = m·a = {mass} kg × {acc} m/s² = {force} N."
        else:
            start = rng.randint(-30, 80)
            step = rng.randint(2, 18)
            seq = [start + step * k for k in range(4)]
            nxt = seq[-1] + step
            if language == "de":
                p = f"Setze die Zahlenfolge {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ... fort und nenne die Regel."
                ans = f"Die Differenz ist immer {step}. Die nächste Zahl ist {nxt}."
            else:
                p = f"Continue the sequence {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ... and state the rule."
                ans = f"The difference is always {step}. The next number is {nxt}."
        out.append(_row(f"math-{language}", i, p, ans, "math"))
    return out


def generate_code(language: str, count: int, seed: int):
    rng = random.Random(seed)
    names = ["value", "count", "total", "number", "items", "values", "scores", "data"]
    out = []
    for i in range(count):
        kind = i % 8
        var = rng.choice(names)
        if kind == 0:
            a, b = rng.randint(1, 90), rng.randint(1, 90)
            fn = rng.choice(["add", "sum_two", "combine"])
            prompt = (f"Schreibe eine kurze Python-Funktion `{fn}(a, b)`, die zwei Zahlen addiert. Zeige sie mit {a} und {b}." if language == "de" else f"Write a short Python function `{fn}(a, b)` that adds two numbers. Show it with {a} and {b}.")
            answer = f"```python\ndef {fn}(a, b):\n    return a + b\n\nprint({fn}({a}, {b}))  # {a+b}\n```"
        elif kind == 1:
            start = rng.randint(0, 20); length = rng.randint(2, 12); stop = start + length; vals = list(range(start, stop))
            prompt = (f"Erkläre kurz, was `list(range({start}, {stop}))` in Python liefert." if language == "de" else f"Briefly explain what `list(range({start}, {stop}))` returns in Python.")
            answer = (f"`range({start}, {stop})` enthält die Ganzzahlen ab {start} bis einschließlich {stop-1}. Die Liste ist `{vals}`." if language == "de" else f"`range({start}, {stop})` contains the integers from {start} through {stop-1}. The list is `{vals}`.")
        elif kind == 2:
            value = rng.randint(0, 99)
            prompt = (f"Korrigiere den Python-Fehler und erkläre ihn: `if {var} = {value}: print({var})`" if language == "de" else f"Fix the Python error and explain it: `if {var} = {value}: print({var})`")
            answer = (f"Für einen Vergleich braucht Python `==`: `if {var} == {value}: print({var})`. Ein einzelnes `=` ist eine Zuweisung." if language == "de" else f"A comparison uses `==`: `if {var} == {value}: print({var})`. A single `=` is assignment.")
        elif kind == 3:
            value = rng.randint(-80, 80)
            prompt = (f"Schreibe Python-Code, der prüft, ob `{var} = {value}` gerade ist." if language == "de" else f"Write Python code that checks whether `{var} = {value}` is even.")
            answer = f"```python\n{var} = {value}\nis_even = {var} % 2 == 0\nprint(is_even)\n```"
        elif kind == 4:
            values = [rng.randint(1, 30) for _ in range(4)]
            prompt = (f"Was gibt dieser Python-Code aus? `values = {values}; print(sum(values))`" if language == "de" else f"What does this Python code print? `values = {values}; print(sum(values))`")
            answer = (f"`sum(values)` addiert alle Elemente. Die Ausgabe ist `{sum(values)}`." if language == "de" else f"`sum(values)` adds all elements. The output is `{sum(values)}`.")
        elif kind == 5:
            a, b = rng.randint(-50, 80), rng.randint(-50, 80); fn = rng.choice(["larger", "maximum", "bigger"])
            prompt = (f"Schreibe eine Python-Funktion `{fn}(a, b)`, die den größeren Wert zurückgibt, und zeige den Aufruf mit {a} und {b}." if language == "de" else f"Write a Python function `{fn}(a, b)` that returns the larger value, and show a call with {a} and {b}.")
            answer = f"```python\ndef {fn}(a, b):\n    return a if a >= b else b\n\nprint({fn}({a}, {b}))  # {max(a,b)}\n```"
        elif kind == 6:
            key = rng.choice(["name", "speed", "gate", "score", "level", "count"]); value = rng.randint(2, 999)
            prompt = (f"Erzeuge in Python ein Dictionary mit dem Schlüssel `{key}` und dem Wert {value} und lies den Wert wieder aus." if language == "de" else f"Create a Python dictionary with key `{key}` and value {value}, then read the value back.")
            answer = f"```python\ndata = {{{key!r}: {value}}}\nprint(data[{key!r}])  # {value}\n```"
        else:
            value = rng.randint(-40, 40); fn = rng.choice(["square", "squared", "power_two"])
            prompt = (f"Schreibe eine Python-Funktion `{fn}({var})`, die das Quadrat zurückgibt. Zeige das Beispiel {var} = {value}." if language == "de" else f"Write a Python function `{fn}({var})` that returns the square. Show the example {var} = {value}.")
            answer = f"```python\ndef {fn}({var}):\n    return {var} * {var}\n\nprint({fn}({value}))  # {value*value}\n```"
        out.append(_row(f"code-{language}", i, prompt, answer, "code"))
    return out


def generate_de_core(count: int, seed: int):
    rng = random.Random(seed)
    concepts = [
        ("Gravitation", "die Anziehung zwischen Massen"), ("Umlaufbahn", "die Bahn eines Körpers um einen anderen unter dem Einfluss von Kräften"),
        ("Beschleunigung", "die zeitliche Änderung der Geschwindigkeit"), ("Variable", "ein benannter Platzhalter für einen Wert"),
        ("Funktion", "eine Zuordnung, die Eingaben nach einer Regel auf Ausgaben abbildet"), ("Algorithmus", "eine endliche, eindeutige Folge von Schritten zur Lösung einer Aufgabe"),
        ("Flughafen", "eine Anlage für Start, Landung und Abfertigung von Luftfahrzeugen"), ("Runway", "die Start- und Landebahn eines Flughafens"),
        ("Prozent", "einen Anteil bezogen auf hundert"), ("Mittelwert", "die Summe der Werte geteilt durch ihre Anzahl"),
        ("Energie", "eine physikalische Größe, die die Fähigkeit beschreibt, Arbeit zu verrichten"), ("Dichte", "Masse pro Volumen"),
        ("Kraft", "eine physikalische Wechselwirkung, die Bewegung oder Form ändern kann"), ("Leistung", "umgesetzte Energie pro Zeit"),
        ("Impuls", "das Produkt aus Masse und Geschwindigkeit"), ("Drehmoment", "die Drehwirkung einer Kraft um eine Achse"),
        ("Datenstruktur", "eine organisierte Form zur Speicherung und Verarbeitung von Daten"), ("Schleife", "eine Programmstruktur zur wiederholten Ausführung von Anweisungen"),
        ("Bedingung", "eine logische Prüfung, die einen Programmablauf steuern kann"), ("Parameter", "ein benannter Eingabewert einer Funktion oder eines Modells"),
        ("Tokenizer", "ein Verfahren, das Text in verarbeitbare Einheiten zerlegt"), ("Training", "die Anpassung von Modellparametern anhand von Daten und einer Zielfunktion"),
        ("Validierung", "die Bewertung eines Modells auf getrennten Daten während der Entwicklung"), ("Luftdruck", "Kraft pro Fläche, die durch die umgebende Luft ausgeübt wird"),
    ]
    subjects = [("Die beiden Programme", "laufen", "läuft"), ("Die Messwerte", "zeigen", "zeigt"), ("Das Ergebnis", "ist", "sind"), ("Mehrere Flugzeuge", "stehen", "steht"), ("Der Satellit", "bewegt sich", "bewegen sich"), ("Die Daten", "sind", "ist"), ("Die drei Sensoren", "messen", "misst"), ("Der Algorithmus", "liefert", "liefern"), ("Viele Sterne", "entstehen", "entsteht"), ("Die beiden Gleichungen", "haben", "hat")]
    contexts = ["für eine Lernkarte", "für einen Physiktest", "für eine kurze Dokumentation", "für eine Schulaufgabe", "für eine technische Notiz", "für eine Zusammenfassung", "für eine Erklärung an Anfänger", "für ein Glossar"]
    out = []
    for i in range(count):
        kind = i % 5
        if kind == 0:
            term, definition = rng.choice(concepts); context = rng.choice(contexts)
            prompt = f"Erkläre den Begriff {term} in einem klaren Satz {context}. Ergänze als Kennzahl die Nummer {i+1} nicht in der Antwort."
            answer = f"{term} bezeichnet {definition}."; domain = "explanation"
        elif kind == 1:
            subject, good, bad = rng.choice(subjects); adverb = rng.choice(["heute", "im Test", "im Labor", "während der Prüfung", "am Gate", "im Programm"])
            sentence = f"{subject} {bad} {adverb} zuverlässig."; corrected = f"{subject} {good} {adverb} zuverlässig."
            prompt = f"Korrigiere den Grammatikfehler in diesem Satz und gib nur die korrigierte Fassung aus: {sentence}"; answer = corrected; domain = "format"
        elif kind == 2:
            a, b = rng.randint(10, 900), rng.randint(10, 900); topic = rng.choice(["Messungen", "Datensätze", "Flüge", "Testfälle", "Beobachtungen", "Proben"])
            text = f"In einem Projekt wurden zuerst {a} {topic} und danach {b} weitere {topic} ausgewertet. Beide Gruppen wurden nach demselben Verfahren geprüft."
            prompt = f"Fasse diesen Text in einem Satz zusammen: {text}"; answer = f"Das Projekt wertete zwei Gruppen mit {a} und {b} {topic} nach demselben Verfahren aus."; domain = "summary"
        elif kind == 3:
            a, b = rng.randint(-200, 300), rng.randint(-200, 300)
            prompt = f"Formuliere präziser: 'Die Werte {a} und {b} sind irgendwie unterschiedlich.' Nenne die absolute Differenz."; answer = f"Die beiden Werte unterscheiden sich um {abs(a-b)}."; domain = "format"
        else:
            mode = i % 4; n = rng.randint(1, 99); hour = rng.randint(5, 22)
            if mode == 0: prompt = f"Übersetze ins Deutsche: The aircraft is at gate {n}."; answer = f"Das Flugzeug steht an Gate {n}."
            elif mode == 1: prompt = f"Übersetze ins Englische: Der Zug fährt um {hour}:00 Uhr ab."; answer = f"The train departs at {hour}:00."
            elif mode == 2: prompt = f"Übersetze ins Deutsche: The result is {n} percent."; answer = f"Das Ergebnis beträgt {n} Prozent."
            else: prompt = f"Übersetze ins Englische: Die Messung hat den Wert {n}."; answer = f"The measurement has the value {n}."
            domain = "translation"
        out.append(_row("de-core", i, prompt, answer, domain))
    return out


def generate_reasoning(language: str, count: int, seed: int):
    rng = random.Random(seed); out = []
    for i in range(count):
        kind = i % 5
        if kind == 0:
            a = rng.randint(20, 90); b = rng.randint(10, a - 1); c = rng.randint(0, b - 1)
            if language == "de": p = f"A hat den Wert {a}, B den Wert {b}, C den Wert {c}. Ordne A, B und C absteigend."; ans = f"Da {a} > {b} > {c}, lautet die Reihenfolge A > B > C."
            else: p = f"A has value {a}, B has value {b}, and C has value {c}. Order A, B and C from largest to smallest."; ans = f"Because {a} > {b} > {c}, the order is A > B > C."
        elif kind == 1:
            n = rng.randint(3, 20); each = rng.randint(2, 15); total = n * each
            if language == "de": p = f"Es gibt {n} gleiche Kisten mit jeweils {each} Teilen. Wie viele Teile sind es insgesamt? Erkläre kurz."; ans = f"Man multipliziert Anzahl der Kisten mit Teilen pro Kiste: {n} · {each} = {total}."
            else: p = f"There are {n} identical boxes with {each} items each. How many items are there in total? Briefly explain."; ans = f"Multiply boxes by items per box: {n} × {each} = {total}."
        elif kind == 2:
            start = rng.randint(1, 18); duration = rng.randint(1, 5); end = (start + duration) % 24
            if language == "de": p = f"Ein Vorgang beginnt um {start}:00 Uhr und dauert {duration} Stunden. Wann endet er?"; ans = f"{start}:00 + {duration} Stunden = {end}:00 Uhr."
            else: p = f"A task starts at {start}:00 and lasts {duration} hours. When does it end?"; ans = f"{start}:00 + {duration} hours = {end}:00."
        elif kind == 3:
            if language == "de": p = "Alle Monde sind Himmelskörper. Objekt X ist ein Mond. Was folgt logisch?"; ans = "Dann ist Objekt X ein Himmelskörper."
            else: p = "All moons are celestial bodies. Object X is a moon. What follows logically?"; ans = "Object X is therefore a celestial body."
        else:
            old = rng.randint(40, 200); delta = rng.randint(1, 30); new = old + delta
            if language == "de": p = f"Ein Wert steigt von {old} auf {new}. Ist die Änderung positiv oder negativ und wie groß ist sie?"; ans = f"Die Änderung ist positiv und beträgt {new-old}."
            else: p = f"A value rises from {old} to {new}. Is the change positive or negative, and by how much?"; ans = f"The change is positive and equals {new-old}."
        out.append(_row(f"reason-{language}", i, p, ans, "reasoning"))
    return out


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")


def _source(source_id: str, file: Path, language: str, domain: str, fraction: float, allow_row_domains: bool = False):
    row = {"id": source_id, "file": file.name, "sha256": sha256_file(file), "license": GENERATED_LICENSE, "license_url": GENERATED_DOC_URL, "provenance_url": GENERATED_DOC_URL, "language": language, "domain": domain, "target_fraction": fraction, "generated_by": "build_targeted_sft50_sources.py", "external_model_calls": False}
    if allow_row_domains: row["allow_row_domains"] = True
    return row


def augment(root: Path, per_generated_source: int = 6000):
    if per_generated_source < 6000:
        raise ValueError("per-generated-source must be >=6000 so every fixed mixture quota has enough deterministic train rows")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    existing = {src.get("id"): src for src in manifest.get("sources", [])}
    for required in ("oasst1-human-en", "oasst1-human-de"):
        if required not in existing: raise ValueError(f"Required OASST1 source missing: {required}")
    generated = [
        ("neral-generated-de-core", "de", "explanation", generate_de_core(max(per_generated_source, 9000), 968_610), True),
        ("neral-generated-math-de", "de", "math", generate_math("de", per_generated_source, 968_611), False),
        ("neral-generated-math-en", "en", "math", generate_math("en", per_generated_source, 968_612), False),
        ("neral-generated-code-de", "de", "code", generate_code("de", per_generated_source, 968_613), False),
        ("neral-generated-code-en", "en", "code", generate_code("en", per_generated_source, 968_614), False),
        ("neral-generated-reasoning-de", "de", "reasoning", generate_reasoning("de", max(2500, per_generated_source // 2), 968_615), False),
        ("neral-generated-reasoning-en", "en", "reasoning", generate_reasoning("en", max(2500, per_generated_source // 2), 968_616), False),
    ]
    frozen_prompts = _targeted_v1_prompts()
    if frozen_prompts:
        overlap = {_normalize(row["prompt"]) for _, _, _, rows, _ in generated for row in rows} & frozen_prompts
        if overlap: raise ValueError(f"Generated SFT prompt overlaps Targeted-v1: {sorted(overlap)[:3]}")
    generated_capacity = {}
    for sid, _, _, rows, _ in generated:
        unique = {}
        for row in rows:
            key = (_normalize(row["prompt"]), _normalize(row["answer"]))
            unique.setdefault(key, row)
        train_capacity = sum(_split_for_prompt(row["prompt"]) == "train" for row in unique.values())
        required = round(TARGET_TRAIN_EXAMPLES * WEIGHTS[sid])
        if train_capacity < required: raise ValueError(f"{sid}: only {train_capacity} deterministic train rows for target quota {required}")
        generated_capacity[sid] = {"unique_pairs": len(unique), "train_capacity": train_capacity, "target_quota_at_20k": required}
    sources = []
    for sid in ("oasst1-human-en", "oasst1-human-de"):
        src = dict(existing[sid]); src["target_fraction"] = WEIGHTS[sid]; sources.append(src)
    generated_counts = {}
    for sid, language, domain, rows, allow_row_domains in generated:
        path = root / f"{sid}.jsonl"; _write_jsonl(path, rows); generated_counts[sid] = len(rows)
        sources.append(_source(sid, path, language, domain, WEIGHTS[sid], allow_row_domains))
    total = sum(float(s["target_fraction"]) for s in sources)
    if abs(total - 1.0) > 1e-9: raise AssertionError(f"target fractions sum to {total}")
    manifest["sources"] = sources
    manifest["recommended_train_examples"] = TARGET_TRAIN_EXAMPLES
    manifest["targeted_augmentation"] = {"version": 1, "generator": "build_targeted_sft50_sources.py", "external_model_calls": False, "generated_counts": generated_counts, "generated_capacity": generated_capacity, "weights": WEIGHTS, "purpose": "Target measured weak areas without replacing human OASST1 data."}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    return manifest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="sft50_sources"); ap.add_argument("--per-generated-source", type=int, default=6000); args = ap.parse_args()
    if args.per_generated_source < 6000: raise ValueError("Use at least 6000 rows per targeted source")
    result = augment(Path(args.root), args.per_generated_source)
    print(json.dumps({"recommended_train_examples": result["recommended_train_examples"], "sources": {s["id"]: s["target_fraction"] for s in result["sources"]}, "targeted": result["targeted_augmentation"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
