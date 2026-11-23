"""V4.3 reference 50: prospective four-category paired suite; CPU-only.

Never reads any Blind200 artifact or historical private prompts. Generates
independent fresh fixtures from a run-local secret, seals SHA256 before GPU
inference and refuses overwrite. The benchmark is not the GPU runner.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import statistics

FROZEN="c0a6be607a0797908e101db460e3ac297a70f088"
COUNTS={"everyday":12,"knowledge":12,"math_logic":14,"programming":12}
ENGINES=("raw_qwen35_9b","neral10_obsidian")
KNOWLEDGE=(
    ("In welchem Jahr wurde die erste Berliner U-Bahn eröffnet?", "1902"),
    ("Welches chemische Element hat das Symbol Fe?", "Eisen"),
    ("Wie viele Planeten hat das Sonnensystem nach heutiger IAU-Klassifikation?", "8"),
    ("Wie heißt die SI-Einheit des elektrischen Widerstands?", "Ohm"),
    ("Wie viele Grad hat ein Innenwinkel eines regelmäßigen Sechsecks?", "120"),
    ("Welche Stadt ist Hauptstadt von Portugal?", "Lissabon"),
    ("Wie heißt das größte Organ des menschlichen Körpers?", "Haut"),
    ("Welches Gas ist mit etwa 78 Prozent Hauptbestandteil trockener Luft?", "Stickstoff"),
    ("Wie viele Meter umfasst ein Kilometer?", "1000"),
    ("Welcher Stern ist das Zentrum unseres Sonnensystems?", "Sonne"),
    ("Welches chemische Element hat die Ordnungszahl 6?", "Kohlenstoff"),
    ("In welchem Jahr fiel die Berliner Mauer?", "1989"),
)
EVERYDAY=(
    "Erkläre den Unterschied zwischen Verdunstung und Sieden an einem Alltagsbeispiel.",
    "Warum beschlägt eine Brille beim Betreten eines warmen Raums?",
    "Wie entsteht Reif auf einer kalten Fensterscheibe?",
    "Warum spart ein geschlossener Kühlschrank Energie?",
    "Warum sollte man ein nasses Handtuch ausgebreitet trocknen?",
    "Erkläre, warum ein Fahrradreifen bei Kälte weniger Druck anzeigt.",
    "Wieso ist Schatten an einem heißen Tag oft angenehmer?",
    "Erkläre, warum Salz Eis auf einer Straße zum Schmelzen bringt.",
    "Warum kann ein Metalllöffel heißer wirken als ein Holzlöffel?",
    "Warum braucht man beim Kochen auf einem Berg oft länger?",
    "Warum flackert eine Kerze im Luftzug?",
    "Erkläre den Unterschied zwischen Masse und Gewicht mit einem Beispiel.",
)


def _record(tag, index, prompt, *, grade, reference=None):
    result={"id":f"v43-ref50-{tag}-{index:02d}","category":tag,
            "prompt":prompt,"history":[],"grader":grade}
    if reference is not None:
        result["sealed_reference"]=reference
    return result


def generate_cases(secret_hex: str) -> list[dict]:
    if type(secret_hex) is not str or re.fullmatch(r"[0-9a-f]{64}",secret_hex) is None:
        raise ValueError("Exact 256-bit secret seed required")
    rng=random.Random(int(secret_hex,16))
    cases=[]
    for idx,theme in enumerate(EVERYDAY):
        variant=rng.choice(("Antworte in zwei bis vier Sätzen.", "Erkläre verständlich ohne Formeln.",
                            "Nenne einen konkreten Grund."))
        cases.append(_record("everyday",idx,theme+" "+variant,grade="blind_pairwise"))
    for idx,(prompt,answer) in enumerate(KNOWLEDGE):
        # Gold values are static and published; independent of model outputs.
        cases.append(_record("knowledge",idx,prompt+" Antworte nur mit dem gesuchten Wort oder der Zahl.",
                     grade="independent_exact",
                     reference={"kind":"string","value":answer}))
    seen=set()
    for idx in range(COUNTS["math_logic"]):
        while True:
            boxes=rng.randint(11,37); each=rng.randint(13,47); removed=rng.randint(9,75)
            key=(boxes,each,removed)
            if key not in seen: seen.add(key);break
        total=boxes*each-removed
        prompt=(f"Ein Depot erhält {boxes} Kisten mit jeweils {each} Teilen und gibt "
                f"{removed} Teile weiter. Wie viele Teile bleiben? Rechne nach; "
                f"letzte Zeile exakt ENDERGEBNIS: <ganze Zahl>.")
        cases.append(_record("math_logic",idx,prompt,grade="independent_exact",
                             reference={"kind":"integer","value":str(total)}))
    seen_programs=set()
    for idx in range(COUNTS["programming"]):
        # A random seed is not permission to duplicate tasks: resample until
        # every coding fixture has a unique, independent specification.
        while True:
            offset=rng.randint(-9,11)
            divisor=rng.randint(2,7)
            excluded=rng.randrange(divisor)
            signature=(offset,divisor,excluded)
            if signature not in seen_programs:
                seen_programs.add(signature)
                break
        def transform(values):
            return sorted(set(v*v+offset for v in values if v%divisor!=excluded))
        tests=[
            {"args":[[]],"expected":[]},
            {"args":[[-5,-2,0,1,2,2,7]],"expected":transform([-5,-2,0,1,2,2,7])},
            {"args":[[-11,3,4,10,11]],"expected":transform([-11,3,4,10,11])},
        ]
        prompt=(f"Schreibe ausschließlich Python-Code: def solve(values):. "
                f"Für jedes Integer v in values verwerfe Werte mit v % {divisor} == {excluded}; "
                f"bilde für die übrigen v*v + ({offset}), entferne Duplikate und "
                f"liefere die aufsteigend sortierte Liste. Leere Eingabe ergibt []. "
                f"Keine externen Pakete.")
        cases.append(_record("programming",idx,prompt,grade="test_execution",
                             reference={"function":"solve","tests":tests}))
    if (len(cases)!=50 or dict(Counter(c["category"] for c in cases))!=COUNTS
            or len({c["id"] for c in cases})!=50
            or len({c["prompt"] for c in cases})!=50):
        raise ValueError("Invalid 50-case generation")
    rng.shuffle(cases)
    return cases


def _encode(cases):
    return "".join(json.dumps(c,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n"
                   for c in cases).encode("utf-8")


def seal_cases(cases, sealed: Path, manifest_path: Path):
    if sealed.exists() or manifest_path.exists():
        raise FileExistsError("Benchmark baseline already present, refuse overwrite")
    if len(cases)!=50 or dict(Counter(c["category"] for c in cases))!=COUNTS:
        raise ValueError("Invalid baseline distribution")
    payload=_encode(cases)
    digest=sha256(payload).hexdigest()
    manifest={"sha256":digest,"count":50,"categories":COUNTS,
              "source_commit":FROZEN,"engines":list(ENGINES),
              "grading":"exact_or_offline_docker; everyday needs blinded human review",
              "no_blind200_reuse":True}
    sealed.parent.mkdir(parents=True,exist_ok=True)
    with sealed.open("xb") as fp: fp.write(payload)
    try:
        with manifest_path.open("x",encoding="utf-8") as fp:
            json.dump(manifest,fp,sort_keys=True,indent=2)
    except Exception:
        sealed.unlink(missing_ok=True)
        raise
    return digest


def verify_seal(sealed: Path, manifest_path: Path):
    if not sealed.is_file() or sealed.is_symlink() or sealed.stat().st_size>100000:
        raise ValueError("Sealed fixture missing or invalid")
    m=json.loads(manifest_path.read_text("utf-8"))
    content=sealed.read_bytes()
    if (sha256(content).hexdigest()!=m.get("sha256")
            or m.get("source_commit")!=FROZEN or m.get("count")!=50
            or m.get("engines")!=list(ENGINES)):
        raise ValueError("Reference suite seal mismatch")
    cases=[json.loads(x) for x in content.decode("utf-8").splitlines() if x]
    if len(cases)!=50 or dict(Counter(x["category"] for x in cases))!=COUNTS:
        raise ValueError("Unbalanced reference suite")
    if len(set(x["id"] for x in cases))!=50 or len({x["prompt"] for x in cases})!=50:
        raise ValueError("Duplicate reference case")
    return cases


def aggregate(cases, records):
    keys={(e,c["id"]) for c in cases for e in ENGINES}
    found=[(x.get("engine"),x.get("id")) for x in records]
    if len(found)!=100 or set(found)!=keys or len(set(found))!=100:
        raise ValueError("Incomplete, duplicated or foreign baseline results")
    by_engine=defaultdict(list)
    for row in records:
        if row.get("grade") not in ("pass","fail","ungraded"):
            raise ValueError("Invalid grade, HTTP200 cannot imply success")
        by_engine[row["engine"]].append(row)
    output={"source_commit":FROZEN,"pairs":50,"observations":100,
            "engines":{},"no_training_capture":True}
    for engine,rows in by_engine.items():
        latency=[int(r["latency_ms"]) for r in rows if type(r.get("latency_ms")) is int]
        vram=[int(r["vram_total_peak_mib"]) for r in rows if type(r.get("vram_total_peak_mib")) is int]
        output["engines"][engine]={
            "passed":sum(r["grade"]=="pass" for r in rows),
            "failed":sum(r["grade"]=="fail" for r in rows),
            "ungraded":sum(r["grade"]=="ungraded" for r in rows),
            "median_latency_ms":statistics.median(latency) if latency else None,
            "gpu_total_peak_mib":max(vram) if vram else None,
            "by_category":{cat:{"pass":sum(r["grade"]=="pass" and r["category"]==cat for r in rows),
                                "fail":sum(r["grade"]=="fail" and r["category"]==cat for r in rows),
                                "ungraded":sum(r["grade"]=="ungraded" and r["category"]==cat for r in rows)}
                           for cat in COUNTS},
        }
    return output
