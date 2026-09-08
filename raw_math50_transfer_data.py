"""Deterministic transfer curriculum for Neral-50M RawMath generalization.

The existing RawMath sampler teaches the frozen benchmark's in-distribution operation families,
but the benchmark's transfer/OOD slice deliberately uses wider ranges. This module adds a
training-only, range-disjoint curriculum: operation families are familiar, concrete operands
are outside both frozen benchmark ID ranges and frozen transfer/OOD ranges wherever applicable.
Exact normalized frozen benchmark/dev prompts are additionally banned.

No external model or corpus is used. This file defines data only; it does not train weights.
"""
from __future__ import annotations

import random
import unicodedata
from fractions import Fraction

from raw_math50_data import CATEGORIES, load_benchmark, make_dev

TRANSFER_TRAIN_SEED = 968_740


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def frozen_prompts() -> set[str]:
    return {normalize(r["prompt"]) for r in load_benchmark() + make_dev()}


def _lp(lang: str, de: str, en: str) -> str:
    return de if lang == "de" else en


def sample_transfer_problem(rng: random.Random, category: str, language: str) -> dict:
    if language not in {"de", "en"}:
        raise ValueError("language must be de/en")
    variant = rng.randrange(4)

    if category == "add_sub":
        a, b = rng.randint(200, 999), rng.randint(200, 999)
        op = rng.choice(["+", "-"])
        ans = a + b if op == "+" else a - b
        de = [f"Berechne {a} {op} {b}. Nur die Zahl.", f"Was ergibt {a} {op} {b}? Antworte nur numerisch.",
              f"Rechne {a} {op} {b} aus; gib ausschließlich das Ergebnis an.", f"Nur Ergebnis: {a} {op} {b}."][variant]
        en = [f"Compute {a} {op} {b}. Number only.", f"What is {a} {op} {b}? Answer numerically only.",
              f"Calculate {a} {op} {b}; return only the result.", f"Result only: {a} {op} {b}."][variant]
        key = f"transfer_train:add_sub:{a}:{op}:{b}"
    elif category == "mul_div":
        if rng.random() < .55:
            a, b = rng.randint(81, 160), rng.randint(81, 160)
            ans = a * b
            de = [f"Berechne {a} × {b}. Nur das Ergebnis.", f"Was ist {a} mal {b}? Nur die Zahl.",
                  f"Multipliziere {a} mit {b}; antworte nur mit dem Resultat.", f"Nur Ergebnis: {a}·{b}."][variant]
            en = [f"Compute {a} × {b}. Result only.", f"What is {a} times {b}? Number only.",
                  f"Multiply {a} by {b}; return only the result.", f"Result only: {a}*{b}."][variant]
            key = f"transfer_train:mul:{a}:{b}"
        else:
            d, q = rng.randint(16, 40), rng.randint(31, 120)
            n = d * q; ans = q
            de = [f"Berechne {n} ÷ {d}. Nur das Ergebnis.", f"Wie viel ist {n} geteilt durch {d}? Nur die Zahl.",
                  f"Teile {n} durch {d}; gib nur den Quotienten an.", f"Nur Ergebnis: {n}/{d}."][variant]
            en = [f"Compute {n} ÷ {d}. Result only.", f"What is {n} divided by {d}? Number only.",
                  f"Divide {n} by {d}; return only the quotient.", f"Result only: {n}/{d}."][variant]
            key = f"transfer_train:div:{n}:{d}"
    elif category == "negatives":
        def wide():
            mag = rng.randint(101, 500)
            return -mag if rng.random() < .5 else mag
        a, b = wide(), wide(); op = rng.choice(["+", "-"])
        ans = a + b if op == "+" else a - b
        de = [f"Berechne {a} {op} ({b}). Nur das Ergebnis.", f"Rechne mit Vorzeichen: {a} {op} ({b}). Nur die Zahl.",
              f"Bestimme {a} {op} ({b}); antworte ausschließlich numerisch.", f"Nur Ergebnis: {a} {op} ({b})."][variant]
        en = [f"Compute {a} {op} ({b}). Result only.", f"Calculate with signs: {a} {op} ({b}). Number only.",
              f"Evaluate {a} {op} ({b}); answer numerically only.", f"Result only: {a} {op} ({b})."][variant]
        key = f"transfer_train:neg:{a}:{op}:{b}"
    elif category == "fractions":
        a, b = rng.randint(13, 40), rng.randint(16, 50)
        c, d = rng.randint(13, 40), rng.randint(16, 50)
        op = rng.choice(["+", "-"])
        f = Fraction(a, b) + (Fraction(c, d) if op == "+" else -Fraction(c, d))
        ans = str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        de = [f"Berechne {a}/{b} {op} {c}/{d} und kürze vollständig. Nur das Ergebnis.",
              f"Addiere oder subtrahiere: {a}/{b} {op} {c}/{d}. Vollständig gekürzt, nur Ergebnis.",
              f"Bestimme den gekürzten Wert von {a}/{b} {op} {c}/{d}. Keine Erklärung.",
              f"Nur gekürztes Ergebnis: {a}/{b} {op} {c}/{d}."][variant]
        en = [f"Compute {a}/{b} {op} {c}/{d} and simplify fully. Result only.",
              f"Add or subtract: {a}/{b} {op} {c}/{d}. Fully simplified, result only.",
              f"Find the simplified value of {a}/{b} {op} {c}/{d}. No explanation.",
              f"Simplified result only: {a}/{b} {op} {c}/{d}."][variant]
        key = f"transfer_train:frac:{a}/{b}:{op}:{c}/{d}"
    elif category == "percent":
        p = rng.choice([12, 30, 45, 60, 90])
        base = rng.randint(4, 100) * 25
        val = Fraction(base * p, 100)
        ans = str(val.numerator) if val.denominator == 1 else f"{val.numerator}/{val.denominator}"
        de = [f"Bestimme {p}% von {base}. Nur das Ergebnis.", f"Wie viel sind {p} Prozent von {base}? Nur die Zahl.",
              f"Berechne den Anteil {p}% von {base}; keine Erklärung.", f"Nur Ergebnis: {p}% von {base}."][variant]
        en = [f"Find {p}% of {base}. Result only.", f"What is {p} percent of {base}? Number only.",
              f"Compute {p}% of {base}; no explanation.", f"Result only: {p}% of {base}."][variant]
        key = f"transfer_train:pct:{p}:{base}"
    elif category == "powers":
        a = rng.randint(101, 200); e = rng.choice([2, 3]); ans = a ** e
        de = [f"Berechne {a}^{e}. Nur das Ergebnis.", f"Was ist {a} hoch {e}? Nur die Zahl.",
              f"Potenziere {a} mit Exponent {e}; antworte nur numerisch.", f"Nur Ergebnis: {a}^{e}."][variant]
        en = [f"Compute {a}^{e}. Result only.", f"What is {a} to the power {e}? Number only.",
              f"Raise {a} to exponent {e}; answer numerically only.", f"Result only: {a}^{e}."][variant]
        key = f"transfer_train:pow:{a}:{e}"
    elif category == "linear":
        a = rng.randint(2, 12); x = rng.randint(81, 160); b = rng.randint(-80, 80); c = a * x + b; ans = x
        de = [f"Löse nach x: {a}x + ({b}) = {c}. Nur x.", f"Welche Zahl x erfüllt {a}x + ({b}) = {c}? Nur die Zahl.",
              f"Bestimme x in {a}x + ({b}) = {c}; keine Erklärung.", f"Nur x: {a}x + ({b}) = {c}."][variant]
        en = [f"Solve for x: {a}x + ({b}) = {c}. x only.", f"Which x satisfies {a}x + ({b}) = {c}? Number only.",
              f"Find x in {a}x + ({b}) = {c}; no explanation.", f"x only: {a}x + ({b}) = {c}."][variant]
        key = f"transfer_train:lin:{a}:{b}:{c}"
    elif category == "sequence":
        start = rng.randint(700, 1600); step = rng.randint(51, 90)
        seq = [start + k * step for k in range(4)]; ans = seq[-1] + step
        seq_text = ", ".join(map(str, seq))
        de = [f"Welche Zahl kommt als Nächstes? {seq_text}, ... Nur die Zahl.", f"Setze die Folge fort: {seq_text}, ... Nur nächster Wert.",
              f"Bestimme das nächste Glied von {seq_text}, ... Keine Erklärung.", f"Nur nächstes Glied: {seq_text}, ..."][variant]
        en = [f"What number comes next? {seq_text}, ... Number only.", f"Continue the sequence: {seq_text}, ... Next value only.",
              f"Find the next term of {seq_text}, ... No explanation.", f"Next term only: {seq_text}, ..."][variant]
        key = f"transfer_train:seq:{start}:{step}"
    elif category == "word":
        a = rng.randint(151, 600); b = rng.randint(81, 180)
        if rng.random() < .5:
            ans = a + b; op = "add"
            de = [f"Lea hat {a} Teile und bekommt {b} dazu. Wie viele hat sie jetzt? Nur die Zahl.",
                  f"Ein Lager enthält {a} Teile, {b} kommen hinzu. Neuer Bestand? Nur Ergebnis.",
                  f"Zu {a} vorhandenen Einheiten kommen {b}. Wie viele sind es insgesamt? Keine Erklärung.",
                  f"Nur Ergebnis: {a} vorhandene plus {b} neue Teile."][variant]
            en = [f"Lea has {a} items and gets {b} more. How many now? Number only.",
                  f"A store has {a} items and receives {b} more. New total? Result only.",
                  f"There are {a} existing units plus {b} new ones. How many total? No explanation.",
                  f"Result only: {a} existing plus {b} new items."][variant]
        else:
            if b > a: a, b = b, a
            ans = a - b; op = "sub"
            de = [f"Lea hat {a} Teile und gibt {b} ab. Wie viele bleiben? Nur die Zahl.",
                  f"Ein Lager hat {a} Teile; {b} werden entnommen. Restbestand? Nur Ergebnis.",
                  f"Von {a} Einheiten gehen {b} weg. Wie viele verbleiben? Keine Erklärung.",
                  f"Nur Ergebnis: {a} Teile minus {b} abgegebene."][variant]
            en = [f"Lea has {a} items and gives away {b}. How many remain? Number only.",
                  f"A store has {a} items; {b} are removed. Remaining total? Result only.",
                  f"From {a} units, {b} leave. How many remain? No explanation.",
                  f"Result only: {a} items minus {b} removed."][variant]
        key = f"transfer_train:word_{op}:{a}:{b}"
    else:
        raise ValueError(category)

    prompt = _lp(language, de, en)
    return {"category": category, "language": language, "prompt": prompt, "answer": str(ans), "canonical": key, "distribution": "transfer_train"}


class TransferTrainingProblemSampler:
    def __init__(self, seed: int = TRANSFER_TRAIN_SEED):
        self.rng = random.Random(seed)
        self.banned_prompts = frozen_prompts()
        self.emitted = 0

    def next(self) -> dict:
        for _ in range(10000):
            category = self.rng.choice(CATEGORIES)
            language = "de" if self.rng.random() < .5 else "en"
            row = sample_transfer_problem(self.rng, category, language)
            if normalize(row["prompt"]) in self.banned_prompts:
                continue
            self.emitted += 1
            return row
        raise RuntimeError("Could not sample a transfer-training item outside frozen prompts")


def validate_transfer_curriculum(samples: int = 5000) -> dict:
    sampler = TransferTrainingProblemSampler()
    frozen = frozen_prompts(); seen = set(); categories = {c: 0 for c in CATEGORIES}; languages = {"de": 0, "en": 0}
    for _ in range(samples):
        row = sampler.next(); n = normalize(row["prompt"])
        if n in frozen:
            raise ValueError("transfer curriculum prompt overlaps frozen benchmark/dev")
        seen.add(n); categories[row["category"]] += 1; languages[row["language"]] += 1
    if any(v == 0 for v in categories.values()):
        raise ValueError("transfer curriculum missed a category")
    return {"samples": samples, "unique_prompts": len(seen), "frozen_prompt_overlap": 0,
            "categories": categories, "languages": languages, "external_model_calls": False}


if __name__ == "__main__":
    import json
    print(json.dumps(validate_transfer_curriculum(), indent=2, ensure_ascii=False))
