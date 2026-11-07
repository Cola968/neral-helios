"""Small train-only arithmetic candidate; no evaluation cases or model loading."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import random

SEED = 20261107
PAIRS_PER_FAMILY = 32
FAMILIES = ("ordered_price", "segmented_fuel")
SYSTEM = "Du bist NERAL. Rechne exakt und zeige die verlangten nachprüfbaren Rechenschritte."
GATES = ["human_semantic_review", "local_qwen_tokenizer_and_completion_mask",
         "all_training_sources_decontamination", "frozen_independent_dev_and_release",
         "explicit_training_authorization"]


def _json(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf8")


def _cent(value: Fraction) -> int:
    if value < 0:
        raise ValueError("Negative price")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _graph(family, facts, variant):
    """Fraction producer; the verifier below uses a separate Decimal engine."""
    nodes, values = [], {k: Fraction(v) for k, v in facts.items()}

    def node(name, op, *args):
        a, b = (values[x] for x in args)
        value = {"add": lambda: a + b, "multiply": lambda: a * b,
                 "divide": lambda: a / b, "discount": lambda: a * (100 - b) / 100,
                 "surcharge": lambda: a * (100 + b) / 100}[op]()
        nodes.append({"id": name, "op": op, "args": list(args), "value": str(value)})
        values[name] = value

    if family == "ordered_price":
        first, last = (("discount", "surcharge") if variant == 0
                       else ("surcharge", "discount"))
        node("first", first, "base_cents", first + "_pct")
        node("with_fee", "add", "first", "fee_cents")
        node("last", last, "with_fee", last + "_pct")
        answer = f"{_cent(values['last']) // 100},{_cent(values['last']) % 100:02d} EUR"
    else:
        node("a_per_min", "divide", "rate_a", "minutes_per_hour")
        node("a_burn", "multiply", "a_per_min", "minutes_a")
        node("b_per_min", "divide", "rate_b", "minutes_per_hour")
        node("b_burn", "multiply", "b_per_min", "minutes_b")
        node("reserve_burn", "multiply", "a_per_min" if variant == 0 else "b_per_min", "reserve_min")
        node("flight", "add", "a_burn", "b_burn")
        node("with_reserve", "add", "flight", "reserve_burn")
        node("total", "add", "with_reserve", "taxi_kg")
        answer = f"{values['total']} kg"
    return nodes, answer


def _prompt(family, f, variant, style):
    introductions = ("Berechne exakt.", "Prüfe die Planung rechnerisch.",
                     "Löse den folgenden Auftrag.", "Bestimme das Ergebnis ohne Zwischenrundung.")
    if family == "ordered_price":
        first, last = ((f"{f['discount_pct']}% Rabatt", f"{f['surcharge_pct']}% Aufschlag")
                       if variant == 0 else
                       (f"{f['surcharge_pct']}% Aufschlag", f"{f['discount_pct']}% Rabatt"))
        task = (f"Startpreis {f['base_cents']} Cent. Reihenfolge: {first}, dann "
                f"{f['fee_cents']} Cent feste Gebühr, zuletzt {last}. Prozentwerte gelten "
                "jeweils für den aktuellen Betrag. Runde ausschließlich den Endbetrag "
                "kaufmännisch auf Cent. Zeige jeden Zwischenbetrag in exakten Cent.")
        fmt = "Betrag mit zwei Nachkommastellen und EUR"
    else:
        reserve_rate = "ersten" if variant == 0 else "zweiten"
        task = (f"Abschnitt A: {f['minutes_a']} Minuten bei {f['rate_a']} kg/h. "
                f"Abschnitt B: {f['minutes_b']} Minuten bei {f['rate_b']} kg/h. "
                f"Reserve {f['reserve_min']} Minuten bei der {reserve_rate} Rate; "
                f"Bodenverbrauch zusätzlich {f['taxi_kg']} kg. 60 Minuten sind eine Stunde. "
                "Zeige die Umrechnung beider Raten in kg/min sowie jeden Teilverbrauch. "
                "Verwende exakte gekürzte Brüche, keine Dezimalnäherungen.")
        fmt = "exakter Wert mit kg"
    return introductions[style] + " " + task + "\nGib CHECK mit Rechenschritten und danach FINAL_ANSWER: " + fmt + "."


def _completion(family, facts, nodes, answer):
    values = {k: str(v) for k, v in facts.items()}
    lines = ["CHECK:"]
    for n in nodes:
        a, b = (values[k] for k in n["args"])
        expr = {"add": f"({a}) + ({b})", "multiply": f"({a}) * ({b})",
                "divide": f"({a}) / ({b})", "discount": f"({a}) * (100 - {b}) / 100",
                "surcharge": f"({a}) * (100 + {b}) / 100"}[n["op"]]
        unit = "Cent" if family == "ordered_price" else ("kg/min" if n["id"].endswith("per_min") else "kg")
        lines.append(f"{n['id']}: {expr} = {n['value']} {unit}")
        values[n["id"]] = n["value"]
    if family == "ordered_price":
        lines.append("Erst jetzt kaufmännisch auf ganze Cent runden und in EUR umrechnen.")
    return "\n".join(lines) + "\nFINAL_ANSWER: " + answer


def _expected_schema(family, variant):
    if family == "ordered_price":
        first, last = (("discount", "surcharge") if variant == 0 else ("surcharge", "discount"))
        return [("first", first, ["base_cents", first + "_pct"]),
                ("with_fee", "add", ["first", "fee_cents"]),
                ("last", last, ["with_fee", last + "_pct"])]
    return [("a_per_min", "divide", ["rate_a", "minutes_per_hour"]),
            ("a_burn", "multiply", ["a_per_min", "minutes_a"]),
            ("b_per_min", "divide", ["rate_b", "minutes_per_hour"]),
            ("b_burn", "multiply", ["b_per_min", "minutes_b"]),
            ("reserve_burn", "multiply", ["a_per_min" if variant == 0 else "b_per_min", "reserve_min"]),
            ("flight", "add", ["a_burn", "b_burn"]),
            ("with_reserve", "add", ["flight", "reserve_burn"]),
            ("total", "add", ["with_reserve", "taxi_kg"])]


def _exact_relation(op, a, b, c):
    """Integer cross-products also reject errors below the Decimal tolerance."""
    an, ad, bn, bd, cn, cd = (a.numerator, a.denominator, b.numerator,
                              b.denominator, c.numerator, c.denominator)
    if op == "add":
        return cn * ad * bd == cd * (an * bd + bn * ad)
    if op == "multiply":
        return cn * ad * bd == cd * an * bn
    if op == "divide":
        return bn != 0 and cn * ad * bn == cd * an * bd
    direction = -1 if op == "discount" else 1
    return cn * ad * bd * 100 == cd * an * (100 * bd + direction * bn)


def verify_case(proof, row):
    """Check graph semantics, every intermediate, final answer and rendered text."""
    family, f, variant = proof["family"], proof["facts"], proof["variant"]
    if family not in FAMILIES or type(variant) is not int or variant not in (0, 1):
        raise ValueError("Unknown family or variant")
    expected_facts = ({"base_cents", "discount_pct", "surcharge_pct", "fee_cents"}
                      if family == "ordered_price" else
                      {"rate_a", "rate_b", "minutes_a", "minutes_b", "reserve_min", "taxi_kg", "minutes_per_hour"})
    if set(f) != expected_facts or any(type(v) is not int or not 0 < v < 100000 for v in f.values()):
        raise ValueError("Invalid numerical facts")
    if family == "ordered_price" and not 0 < f["discount_pct"] < 100:
        raise ValueError("Invalid discount")
    if family == "segmented_fuel" and (f["minutes_per_hour"] != 60 or
            any(f[k] % 60 == 0 for k in ("rate_a", "rate_b"))):
        raise ValueError("Trivial or invalid hourly rate conversion")
    nodes = proof["nodes"]
    if [(n["id"], n["op"], n["args"]) for n in nodes] != _expected_schema(family, variant):
        raise ValueError("Operation graph contradicts the task")
    with localcontext() as ctx:
        ctx.prec = 70
        values = {k: Decimal(v) for k, v in f.items()}
        exact_values = {k: Fraction(v) for k, v in f.items()}
        for n in nodes:
            a, b = (values[k] for k in n["args"])
            computed = {"add": lambda: a + b, "multiply": lambda: a * b,
                        "divide": lambda: a / b, "discount": lambda: a * (1 - b / 100),
                        "surcharge": lambda: a * (1 + b / 100)}[n["op"]]()
            claimed = Fraction(n["value"])
            if not _exact_relation(n["op"], *(exact_values[k] for k in n["args"]), claimed):
                raise ValueError("Intermediate fails exact integer relation")
            if abs(computed - Decimal(claimed.numerator) / Decimal(claimed.denominator)) > Decimal("1e-55"):
                raise ValueError("Intermediate fails independent Decimal replay")
            values[n["id"]] = computed
            exact_values[n["id"]] = claimed
        if family == "ordered_price":
            final = values["last"].quantize(Decimal(1), rounding=ROUND_HALF_UP)
            answer = f"{final / 100:.2f} EUR".replace(".", ",")
        else:
            selected = f["rate_a"] if variant == 0 else f["rate_b"]
            direct = (Decimal(f["rate_a"] * f["minutes_a"]) +
                      Decimal(f["rate_b"] * f["minutes_b"]) +
                      Decimal(selected * f["reserve_min"])) / 60 + f["taxi_kg"]
            claimed = Fraction(proof["answer"].removesuffix(" kg"))
            if (claimed != exact_values["total"] or claimed.denominator == 1 or
                    abs(direct - Decimal(claimed.numerator) / claimed.denominator) > Decimal("1e-55")):
                raise ValueError("Final fuel amount must be correct and nonintegral")
            answer = str(claimed) + " kg"
    if proof["answer"] != answer:
        raise ValueError("Wrong final answer or rounding")
    style = proof["style"]
    if type(style) is not int or not 0 <= style < 4:
        raise ValueError("Invalid prompt style")
    expected = {"messages": [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": _prompt(family, f, variant, style)},
        {"role": "assistant", "content": _completion(family, f, nodes, answer)}]}
    if row != expected:
        raise ValueError("Training text contradicts verified proof")
    return True


def candidate(seed=SEED):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Invalid seed")
    rng, rows, proofs = random.Random(seed), [], []
    for family in FAMILIES:
        for index in range(PAIRS_PER_FAMILY):
            for attempt in range(1000):
                facts = ({"base_cents": rng.randint(350, 57000), "discount_pct": rng.randint(3, 44),
                          "surcharge_pct": rng.randint(3, 37), "fee_cents": rng.randint(65, 470)}
                         if family == "ordered_price" else
                         {"rate_a": rng.randint(301, 2137), "rate_b": rng.randint(341, 2371),
                          "minutes_a": rng.randint(14, 173), "minutes_b": rng.randint(17, 191),
                          "reserve_min": rng.randint(15, 79), "taxi_kg": rng.randint(10, 170),
                          "minutes_per_hour": 60})
                pair = []
                try:
                    for variant in (0, 1):
                        nodes, answer = _graph(family, facts, variant)
                        proof = {"id": f"{family}-{index:03d}-{variant}", "pair_id": f"{family}-{index:03d}",
                                 "family": family, "facts": facts, "variant": variant, "style": index % 4,
                                 "nodes": nodes, "answer": answer}
                        row = {"messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": _prompt(family, facts, variant, index % 4)},
                            {"role": "assistant", "content": _completion(family, facts, nodes, answer)}]}
                        verify_case(proof, row)
                        pair.append((proof, row))
                    if pair[0][0]["answer"] == pair[1][0]["answer"]:
                        continue
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("Could not construct a valid distinct counterfactual pair")
            for proof, row in pair:
                proofs.append(proof)
                rows.append(row)
    return rows, proofs


def build(out: Path, seed=SEED):
    out = Path(out)
    if out.exists() or out.is_symlink() or not out.parent.is_dir() or out.parent.is_symlink():
        raise ValueError("New separate local output directory required")
    rows, proofs = candidate(seed)
    payloads = {"train.jsonl": b"".join(map(_json, rows)), "proofs.jsonl": b"".join(map(_json, proofs))}
    manifest = {"schema": 1, "status": "TRAIN_ONLY_ARITHMETIC_CANDIDATE_NOT_TRAINED", "seed": seed,
                "rows": len(rows), "counterfactual_pairs": len(rows) // 2, "families": list(FAMILIES),
                "splits": ["train"], "sha256": {k: sha256(v).hexdigest() for k, v in payloads.items()},
                "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "decimal_replay_verified": True, "training_allowed": False, "training_performed": False,
                "human_quality_reviewed": False, "tokenizer_verified": False, "decontamination_verified": False,
                "general_improvement_proven": False, "pending_gates": GATES}
    if len(payloads["train.jsonl"]) >= 1_048_576:
        raise ValueError("Existing pilot trainer's 1 MiB data limit exceeded")
    out.mkdir()
    for name, data in {**payloads, "manifest.json": _json(manifest)}.items():
        (out / name).write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=SEED, type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
