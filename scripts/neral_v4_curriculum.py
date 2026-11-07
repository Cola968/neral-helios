"""Deterministic NERAL v4 SFT candidate: 2,560 local CPU-verified examples.

No network, GPU, model inference, user data, or training. NOT APPROVED FOR
TRAINING: tokenizer audit and human review are still required. Development
validation and test sets are NOT independent external benchmarks.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
import random
import sqlite3
from pathlib import Path

SEED = 20261107
PER_TEMPLATE = 40
CATEGORIES = ("linear", "both_sides", "fractions", "percent",
              "timezone", "fuel", "python_debug", "sql")
STYLES = (
    "Aufgabe: {task}\nGib ausschließlich {fmt} zurück.",
    "Bitte löse: {task}\nErwartetes Format: {fmt}.",
    "Analysiere: {task}\nAntworte nur mit {fmt}.",
    "Arbeitsauftrag: {task}\nKeine Erklärung; Ausgabe als {fmt}.",
    "Task: {task}\nReturn only {fmt}.",
    "Complete this: {task}\nOutput format: {fmt}.",
    "Prüfe die Aufgabenstellung: {task}\nNur {fmt} angeben.",
    "Independent phrasing: {task}\nRespond with {fmt} and no commentary.",
)
TOTAL = len(CATEGORIES) * len(STYLES) * PER_TEMPLATE
ROWS = {"train": 1920, "val": 320, "test": 320}
SYSTEM = ("Du bist NERAL. Löse überprüfbare technische Aufgaben sachlich. "
          "Beachte ausdrücklich verlangte Antwortformate.")


def _rng(seed: int, category: str, style: int, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{category}:{style}:{index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _signed(n: int) -> str:
    return f"+ {n}" if n >= 0 else f"- {-n}"


def _sql_verified(query: str, *, grouped: bool, floor: int, minimum: int = 0):
    """Independently evaluate generated fixed SELECT on a synthetic in-memory DB."""
    with sqlite3.connect(":memory:") as db:
        if grouped:
            db.execute("CREATE TABLE Track(GenreID INTEGER)")
            data = [g for g in range(1, 61) for _ in range(g % 7 + 1)]
            db.executemany("INSERT INTO Track VALUES (?)", ((x,) for x in data))
            expected = [(k, count) for k, count in sorted(Counter(
                g for g in data if g >= floor).items()) if count >= minimum]
        else:
            db.execute("CREATE TABLE Artist(ArtistID INTEGER, Name TEXT)")
            db.execute("CREATE TABLE Album(AlbumID INTEGER, ArtistID INTEGER)")
            db.executemany("INSERT INTO Artist VALUES (?, ?)",
                           ((i, f"Artist_{i:02}") for i in range(1, 61)))
            db.executemany("INSERT INTO Album VALUES (?, ?)",
                           ((i, i) for i in range(2, 61, 2)))
            expected = [(f"Artist_{i:02}",) for i in range(max(1, floor), 61, 2)
                        if i % 2 == 1]
            # Odd-floor normalization is explicit, so parity is checked.
            expected = [(f"Artist_{i:02}",) for i in range(max(1, floor), 61)
                        if i % 2 == 1]
        db.execute("PRAGMA query_only=ON")
        actual = db.execute(query).fetchall()
        if actual != expected:
            raise ValueError("SQL semantic verification rejected answer")


def make_case(category: str, style: int, index: int, seed: int = SEED) -> dict:
    if (category not in CATEGORIES or type(style) is not int or not 0 <= style < 8
            or type(index) is not int or not 0 <= index < PER_TEMPLATE
            or type(seed) is not int or not 0 <= seed <= 2**63-1):
        raise ValueError("Invalid example coordinates")
    rng = _rng(seed, category, style, index)
    # One independent numerical coordinate per template slot prevents\n    # accidental duplicate prompts in otherwise random synthetic families.\n    if category == "linear":
        a, x, b = index+2, rng.randint(-50,50), rng.randint(-90,90)
        rhs = a*x+b
        task = f"Bestimme x: {a}x {_signed(b)} = {rhs}."
        answer, fmt = f"x = {x}", "x = Zahl"
        assert a*x+b == rhs
    elif category == "both_sides":
        a, d = index+3, rng.randint(1,12)
        if a == d:
            a += 1
        x, b = rng.randint(-45,45), rng.randint(-80,80)
        c = (a-d)*x + b
        task = f"Löse: {a}x {_signed(b)} = {d}x {_signed(c)}."
        answer, fmt = f"{a-d}x = {c-b}\nx = {x}", "zwei Rechenzeilen"
        assert a*x+b == d*x+c and a != d
    elif category == "fractions":
        a, b = index-20, rng.randint(2,17)
        c, d = rng.randint(-22,22), rng.randint(2,19)
        op = ("+", "-", "*", "/")[index % 4]
        if op == "/" and c == 0:
            c = 3
        left, right = Fraction(a,b), Fraction(c,d)
        # Do not eagerly evaluate unused operators (division by zero is
        # invalid even for a multiplication-only generated example).
        if op == "+":
            result = left + right
        elif op == "-":
            result = left - right
        elif op == "*":
            result = left * right
        else:
            result = left / right
        if op == "+":
            assert result-right == left
        elif op == "-":
            assert result+right == left
        elif op == "/" :
            assert result*right == left
        task = f"Berechne exakt: ({a}/{b}) {op} ({c}/{d})."
        answer, fmt = str(result), "gekürzten Bruch oder ganze Zahl"
    elif category == "percent":
        price = (index+1)*100
        discount, surcharge = rng.randint(5,35), rng.randint(3,21)
        cents = price*(100-discount)*(100+surcharge)//100
        assert cents % 1 == 0
        assert Fraction(cents,100) == (Fraction(price)*(100-discount)*(100+surcharge)/10000)
        task = (f"Artikelpreis {price} EUR; erst {discount}% Rabatt, "
                f"dann {surcharge}% Zuschlag auf den reduzierten Preis. Endpreis?")
        answer, fmt = f"{cents//100},{cents%100:02d} EUR", "EUR-Betrag mit zwei Dezimalstellen"
    elif category == "timezone":
        hour, minute = rng.randint(0,23), 15*rng.randint(0,3)
        src, dest = rng.randint(-7,5), rng.randint(-2,11)
        duration = (index+2)*30
        total = hour*60+minute+duration+60*(dest-src)
        day, clock = divmod(total,1440)
        assert total == 1440*day+clock
        task = (f"Abflug {hour:02d}:{minute:02d} bei fest UTC{src:+d}, "
                f"Flugdauer {duration} Minuten, Ziel fest UTC{dest:+d}. "
                "Ankunftszeit und Tagesverschiebung? Keine Sommerzeit.")
        answer, fmt = f"{clock//60:02d}:{clock%60:02d} (Tag {day:+d})", "HH:MM (Tag +/-N)"
    elif category == "fuel":
        burn = rng.randint(10,36)*60
        flight = (index+8)*15
        reserve = rng.randint(2,8)*15
        taxi = rng.randint(15,140)
        value = burn*(flight+reserve)//60+taxi
        assert value*60 == burn*(flight+reserve)+taxi*60
        task = (f"Flugzeug verbraucht {burn} kg/h. Flugdauer {flight} min, "
                f"Reserve {reserve} min, Taxi {taxi} kg. Gesamttreibstoff?")
        answer, fmt = f"{value} kg", "Kilogrammwert"
    elif category == "python_debug":
        name = f"adjust_{style}_{index}"
        multiplier, offset = rng.randint(2,15), rng.randint(1,75)
        bad = f"def {name}(value):\n    return value * {multiplier} - {offset}\n"
        fixed = f"def {name}(value):\n    return value * {multiplier} + {offset}\n"
        task = (f"Korrigiere den Vorzeichenfehler, Ziel ist "
                f"value*{multiplier}+{offset}:\n{bad}")
        answer, fmt = fixed, "eine Python-Funktionsdefinition"
        tree = ast.parse(answer)
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef) and fn.name == name
        assert len(fn.body) == 1 and isinstance(fn.body[0], ast.Return)
        assert isinstance(fn.body[0].value,ast.BinOp) and isinstance(fn.body[0].value.op,ast.Add)
    elif category == "sql":
        floor = index+1
        if index % 2 == 0:
            minimum = rng.randint(2,6)
            task = (f"SQL Track(GenreID): Nur GenreID >= {floor}, "
                    f"gruppieren und nur Gruppen mit mindestens {minimum} Zeilen zeigen, aufsteigend.")
            answer = (f"SELECT GenreID, COUNT(*) AS n FROM Track WHERE GenreID >= {floor} "
                      f"GROUP BY GenreID HAVING COUNT(*) >= {minimum} ORDER BY GenreID ASC;")
            _sql_verified(answer, grouped=True, floor=floor, minimum=minimum)
        else:
            task = (f"SQL Artist(ArtistID,Name), Album(AlbumID,ArtistID): "
                    f"Zeige Namen von Künstlern ohne Album mit ArtistID >= {floor}; sortiere nach Name.")
            answer = ("SELECT Artist.Name FROM Artist LEFT JOIN Album "
                      "ON Artist.ArtistID = Album.ArtistID "
                      f"WHERE Album.AlbumID IS NULL AND Artist.ArtistID >= {floor} "
                      "ORDER BY Artist.Name ASC;")
            _sql_verified(answer,grouped=False,floor=floor)
        fmt = "eine SELECT-Abfrage"
    else:
        raise ValueError("Unknown category")
    prompt = STYLES[style].format(task=task,fmt=fmt)
    return {"prompt":[{"role":"system","content":SYSTEM},
                      {"role":"user","content":prompt}],
            "completion":[{"role":"assistant","content":answer}]}


def _generated(seed: int) -> dict[str,bytes]:
    splits = {"train":[],"val":[],"test":[]}
    seen = set()
    for cat in CATEGORIES:
        for style in range(8):
            split = "train" if style < 6 else ("val" if style == 6 else "test")
            for i in range(PER_TEMPLATE):
                row = make_case(cat,style,i,seed)
                signature = " ".join(row["prompt"][1]["content"].casefold().split())
                if signature in seen:
                    raise ValueError("Duplicate prompt across splits")
                seen.add(signature)
                splits[split].append(json.dumps(row,ensure_ascii=False,
                    sort_keys=True,separators=(",",":"))+"\n")
    if len(seen) != TOTAL:
        raise ValueError("Incomplete generated examples")
    for split, expected in ROWS.items():
        if len(splits[split]) != expected:
            raise ValueError("Unbalanced generated split")
    return {key:"".join(value).encode("utf8") for key,value in splits.items()}


def _manifest(seed: int, payloads: dict[str,bytes]) -> dict:
    return {
        "schema_version":4,
        "dataset":"NERAL v4 verified-template candidate",
        "seed":seed,"categories":list(CATEGORIES),
        "rows":dict(ROWS),
        "sha256":{split:hashlib.sha256(data).hexdigest() for split,data in payloads.items()},
        "split_policy":"Template-group-disjoint: six train styles, one val, one test.",
        "answer_validation":"Original deterministic arithmetic, AST and SQLite oracles",
        "answers_verified":True,
        "training_allowed":False,
        "tokenizer_validated":False,
        "training_performed":False,
        "quality_improvement_proven":False,
        "independent_external_holdout":False,
        "limitations":[
          "Synthetic template data; semantic generalization cannot be inferred.",
          "Validation and test share generators and must not be called independent benchmarks.",
          "No per-item human review or actual Qwen tokenizer preflight yet.",
          "Holdout rows must never be used as training examples."
        ]
    }


def build(root:Path,seed:int=SEED) -> dict:
    if type(seed) is not int or not 0 <= seed <= 2**63-1:
        raise ValueError("Invalid seed")
    root=Path(root)
    if root.is_symlink() or (root.exists() and any(root.iterdir())):
        raise ValueError("Output directory must be new, empty and non-symlinked")
    payloads=_generated(seed)
    root.mkdir(parents=True,exist_ok=True)
    for split,data in payloads.items():
        (root/(split+".jsonl")).write_bytes(data)
    manifest=_manifest(seed,payloads)
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,
        ensure_ascii=False,sort_keys=True)+"\n",encoding="utf8")
    return manifest


def verify(root:Path) -> dict:
    root=Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing/symlinked dataset directory")
    files=set(x.name for x in root.iterdir())
    if files!={"train.jsonl","val.jsonl","test.jsonl","manifest.json"}:
        raise ValueError("Unexpected/missing dataset files")
    if any(p.is_symlink() or not p.is_file() for p in root.iterdir()):
        raise ValueError("Linked or non-file inputs are prohibited")
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf8"))
    seed=manifest.get("seed")
    if type(seed) is not int or not 0 <= seed <= 2**63-1:
        raise ValueError("Missing deterministic seed")
    expected=_generated(seed)
    official=_manifest(seed,expected)
    if manifest != official:
        raise ValueError("Dataset manifest differs from generated contract")
    for split,payload in expected.items():
        path=root/(split+".jsonl")
        if path.stat().st_size>3_000_000 or path.read_bytes()!=payload:
            raise ValueError("Dataset content/label differs from verified oracle")
    return {"schema":4,"rows":dict(ROWS),"answers_verified":True,
            "valid":True,"training_allowed":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=SEED)
    args=parser.parse_args(argv)
    result=build(args.out,args.seed)
    assert verify(args.out)["valid"]
    print(json.dumps({k:result[k] for k in
                      ("dataset","rows","sha256","training_allowed")},sort_keys=True))


if __name__=="__main__":
    main()
