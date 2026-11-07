"""NERAL v6 CPU-only *candidate* for original multi-step transfer examples.

Generates train/validation and isolated synthetic template-shift holdout.
No model loading, network, weight updates, benchmark answers, or training approval.
Synthetic holdout does NOT prove independent real-world generalization.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import random

FAMILIES = ("money_order", "two_leg_fuel", "timezone_stop", "inventory_flow", "mixture_percent")
ROWS = {"train": 600, "validation": 100, "holdout": 100}
PER_FAMILY = {name: count // len(FAMILIES) for name, count in ROWS.items()}
SYSTEM = ("Du bist NERAL AI. Löse die gestellte Aufgabe exakt, prüfe die "
          "Einheiten und halte das Antwortformat ein. Erfinde keine Werkzeuge.")
GENERATOR_ID = "NERAL-v6-original-programs-20261107-v1"


def _rng(family: str, split: str, index: int) -> random.Random:
    seed = sha256(f"{GENERATOR_ID}|{family}|{split}|{index}".encode()).digest()
    return random.Random(int.from_bytes(seed[:16], "big"))


def _cent_half_up(amount: Fraction) -> int:
    if amount < 0:
        raise ValueError("Negative amount")
    return (2 * amount.numerator + amount.denominator) // (2 * amount.denominator)


def _problem(family: str, split: str, index: int) -> tuple[str, str, str]:
    r = _rng(family, split, index)
    if family == "money_order":
        cents = r.randint(350, 57000)
        rise, rebate, fee = r.randint(3, 37), r.randint(5, 44), r.randint(65, 470)
        reverse = index % 2 == 1
        if reverse:
            exact = (Fraction(cents * (100 - rebate), 100) + fee) * Fraction(100 + rise, 100)
            actions = f"zunächst {rebate}% Rabatt, dann eine feste Gebühr von {fee} Cent und zuletzt {rise}% Aufschlag"
            formula = f"(({cents}×{100-rebate}/100)+{fee})×{100+rise}/100"
        else:
            exact = (Fraction(cents * (100 + rise), 100) + fee) * Fraction(100 - rebate, 100)
            actions = f"zunächst {rise}% Aufschlag, dann eine feste Gebühr von {fee} Cent und zuletzt {rebate}% Rabatt"
            formula = f"(({cents}×{100+rise}/100)+{fee})×{100-rebate}/100"
        rounded = _cent_half_up(exact)
        answer = f"{rounded // 100},{rounded % 100:02d} EUR"
        amount = f"{cents // 100},{cents % 100:02d} EUR"
        prompts = {
            "train": f"Ein Angebot startet bei {amount}. Berechne {actions}. Zwischenwerte bleiben exakt; runde erst ganz am Ende kaufmännisch auf Cent. Nenne den Endbetrag.",
            "validation": f"Auf einer Rechnung beträgt der Ursprungspreis {amount}. Der Händler wendet nacheinander diese Regeln an: {actions}. Welche Summe ist nach einmaliger Schlussrundung fällig?",
            "holdout": f"Ein Kauf hat den Listenpreis {amount}. Es gelten in dieser Reihenfolge: {actions}. Es darf erst nach allen Änderungen auf zwei Dezimalstellen gerundet werden. Berechne den Zahlbetrag.",
        }
        check = f"CHECK: {formula} = {exact} Cent; gerundet {rounded} Cent"
    elif family == "two_leg_fuel":
        rate_a, rate_b = 60 * r.randint(5, 36), 60 * r.randint(5, 39)
        t_a, t_b, reserve, taxi = r.randint(14, 200), r.randint(17, 210), r.randint(15, 90), r.randint(10, 170)
        fuel = Fraction(rate_a * t_a + rate_b * (t_b + reserve), 60) + taxi
        answer = f"{fuel} kg"
        prompts = {
            "train": f"Zwei Flugabschnitte: {t_a} min zu {rate_a} kg/h, danach {t_b} min zu {rate_b} kg/h. Eine Reserve von {reserve} min nutzt die zweite Rate, zusätzlich {taxi} kg am Boden. Berechne die Gesamtmasse.",
            "validation": f"Ein Flugzeug verbraucht erst {rate_a} kg pro Stunde für {t_a} Minuten und dann {rate_b} kg pro Stunde für {t_b} Minuten. Für {reserve} Minuten Reserve gilt der zweite Verbrauch, dazu kommen {taxi} kg Taxi-Fuel. Wie groß ist die exakte Summe?",
            "holdout": f"Flugplanung mit zwei Phasen: Phase 1 ({t_a} Minuten, {rate_a} kg/h), Phase 2 ({t_b} Minuten, {rate_b} kg/h). Zusatzreserve {reserve} Minuten bei Rate 2 und fester Bodenverbrauch {taxi} kg. Wie viele kg sind notwendig?",
        }
        check = f"CHECK: {rate_a}×{t_a}/60 + {rate_b}×({t_b}+{reserve})/60 + {taxi} = {fuel} kg"
    elif family == "timezone_stop":
        hour, minute = r.randrange(24), r.choice((0, 15, 30, 45))
        src, dst = r.randint(-10, 10), r.randint(-11, 11)
        travel, pause = r.randint(40, 1150), r.randint(0, 200)
        absolute = hour * 60 + minute + travel + pause + (dst - src) * 60
        day, local = divmod(absolute, 1440)
        answer = f"{local//60:02d}:{local%60:02d} (Tag {day:+d})"
        prompts = {
            "train": f"Reisebeginn {hour:02d}:{minute:02d} Uhr in UTC{src:+d}, reine Reisezeit {travel} min und Aufenthalt {pause} min. Ankunftsregion UTC{dst:+d}. Keine Sommerzeit. Gib Zieluhrzeit und Tagesverschiebung an.",
            "validation": f"Bei festen Zeitzonen startet eine Fahrt um {hour:02d}:{minute:02d} Ortszeit (UTC{src:+d}). Unterwegs vergehen {travel} min plus {pause} min Pause. Die Zielzone ist UTC{dst:+d}. Bestimme die lokale Ankunft inklusive Tag.",
            "holdout": f"Ein internationaler Umlauf beginnt {hour:02d}:{minute:02d} in der Zeitzone UTC{src:+d}; unterwegs sind {travel} Minuten Fahrzeit und {pause} Minuten Umsteigezeit nötig. Ziel UTC{dst:+d}. Ohne DST: Wann ist lokal Ankunft?",
        }
        check = f"CHECK: lokale Zielminute={absolute}; Tag {day:+d}; Restminute={local}"
    elif family == "inventory_flow":
        crates, per = r.randint(3, 40), r.randint(7, 58)
        delivery, returns, damaged = r.randint(4, 88), r.randint(1, 45), r.randint(2, 33)
        total = crates * per + delivery + returns - damaged
        answer = f"{total} Stück"
        prompts = {
            "train": f"Ein Lager enthält {crates} Kartons zu je {per} Teilen. Es kommen {delivery} neue Teile und {returns} zurückgegebene Teile hinzu; {damaged} sind nicht brauchbar. Nenne den verfügbaren Bestand.",
            "validation": f"Inventur: {crates} Kisten mit jeweils {per} Einheiten, Wareneingang {delivery}, Rücklauf {returns}, Ausschuss {damaged}. Wie viele nutzbare Einheiten bleiben?",
            "holdout": f"Eine Werkstatt zählt {crates} Gebinde mit {per} Einheiten pro Gebinde. Hinzu kommen {delivery} gelieferte und {returns} retournierte Einheiten; {damaged} werden aussortiert. Bestimme den nutzbaren Endbestand.",
        }
        check = f"CHECK: {crates}×{per}+{delivery}+{returns}-{damaged}={total} Stück"
    else:
        a, b, water = r.randint(55, 470), r.randint(40, 430), r.randint(0, 300)
        p, q = r.randint(12, 85), r.randint(4, 75)
        percent = Fraction(a * p + b * q, a + b + water)
        answer = f"{percent} %"
        prompts = {
            "train": f"Mische {a} ml einer {p}%igen Lösung und {b} ml einer {q}%igen Lösung. Zusätzlich {water} ml wirkstofffreies Wasser. Wie hoch ist die exakte Konzentration?",
            "validation": f"Ein Gemisch besteht aus {a} ml mit {p}% Wirkstoff, {b} ml mit {q}% Wirkstoff und {water} ml reinem Wasser. Berechne den endgültigen Prozentwert ohne Rundung.",
            "holdout": f"Aus zwei Flüssigkeiten ({a} ml zu {p}% und {b} ml zu {q}%) wird mit {water} ml Wasser eine Mischung erstellt. Welchen Wirkstoffanteil hat das Endvolumen exakt?",
        }
        check = f"CHECK: ({a}×{p}+{b}×{q})/({a}+{b}+{water})={percent} %"
    question = prompts[split]
    if family == "timezone_stop":
        question += " Ausgabeformat HH:MM (Tag +/-N)."
    elif family == "mixture_percent":
        question += " Antworte als vollständig gekürzter Bruch oder Zahl mit %."
    elif family == "inventory_flow":
        question += " Antworte mit Stück."
    elif family == "two_leg_fuel":
        question += " Antworte exakt mit kg."
    else:
        question += " Antworte mit Betrag, zwei Nachkommastellen und EUR."
    return question, answer, check


def make_case(family: str, split: str, index: int) -> dict:
    if (family not in FAMILIES or split not in ROWS or type(index) is not int
            or not 0 <= index < PER_FAMILY[split]):
        raise ValueError("Unexpected family, split or index")
    question, answer, check = _problem(family, split, index)
    if index % 3 == 0:
        question += " Gib zuerst CHECK: Prüfung und dann FINAL_ANSWER: Ergebnis an."
        response = f"{check}\nFINAL_ANSWER: {answer}"
    else:
        question += " Gib ausschließlich das Endergebnis aus."
        response = answer
    return {"family": family, "oracle_sha256": sha256(
                (question + "\n" + answer).encode("utf-8")).hexdigest(),
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]}


def generate() -> tuple[dict[str, bytes], dict]:
    data = {}
    seen = set()
    for split in ROWS:
        rows = []
        for family in FAMILIES:
            for index in range(PER_FAMILY[split]):
                row = make_case(family, split, index)
                prompt = row["messages"][1]["content"].casefold().strip()
                if prompt in seen:
                    raise ValueError("Train/validation/holdout overlap")
                seen.add(prompt)
                rows.append(row)
        data[split] = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    return data, {
        "schema": 1, "status": "NERAL_V6_COMPOSITION_CANDIDATE_NOT_TRAINED",
        "source": "scripts/neral_v6_compositional.py",
        "families": list(FAMILIES), "counts": dict(ROWS),
        "sha256": {k: sha256(v).hexdigest() for k, v in data.items()},
        "training_allowed": False, "training_performed": False,
        "human_review_approved": False, "tokenizer_verified": False,
        "independent_external_benchmark": False,
        "holdout_never_for_training": True,
        "general_improvement_proven": False,
        "limitations": "Only synthetic new templates and seeds; not independent unknown-task evidence.",
    }


def build(out: Path) -> dict:
    out = Path(out)
    if (out.is_symlink() or out.exists() or not out.parent.is_dir()
            or out.parent.is_symlink()):
        raise ValueError("Output must be a new, separate directory")
    data, manifest = generate()
    out.mkdir(exist_ok=False)
    for split, payload in data.items():
        (out / (split + ".jsonl")).write_bytes(payload)
    (out / "manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.out)
    print(json.dumps({"counts": summary["counts"], "sha256": summary["sha256"],
                      "training_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
