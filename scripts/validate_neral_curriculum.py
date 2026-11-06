"""CPU-only NERAL synthetic QLoRA data contract and answer audit.

Proves exact three-message schema, split isolation and deterministic label
correctness. It does NOT establish semantic diversity, tokenizer fit or model
quality. No external downloads or private prompts are used.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import statistics
from pathlib import Path

SYSTEM = ("Du bist NERAL, ein lokaler Coding- und Analyse-Assistent. "
          "Rechne nachvollziehbar, erfinde keine Daten, fuehre keine Aktionen "
          "ohne Freigabe aus und gib klare Antworten.")
CATEGORIES = ("algebra","budget","fuel","timezones","sql_join","sql_group",
              "python_debug","tool_choice","analysis","physics")
SPLITS = ("train","val","test")
EXPECTED = {"train":800,"val":100,"test":100}
MAX_JSONL_BYTES = 1_500_000

def category(user: str) -> str:
    if user.startswith("Loese "): return "algebra"
    if user.startswith("Ein Produkt kostet "): return "budget"
    if user.startswith("Ein Flugzeug verbraucht "): return "fuel"
    if user.startswith("Abflug "): return "timezones"
    if user.startswith("SQL: Tabelle "): return "sql_group"
    if user.startswith("SQL: "): return "sql_join"
    if user.startswith("Korrigiere "): return "python_debug"
    if user.startswith("NERAL-Agent:"): return "tool_choice"
    if user.startswith("Berechne "): return "analysis"
    if user.startswith("Physik:"): return "physics"
    raise ValueError("Unrecognized curriculum question type")

def value(pattern: str, txt: str) -> tuple[str, ...]:
    found = re.search(pattern,txt)
    if not found:
        raise ValueError("Missing required numerical evidence")
    return found.groups()

def audit_answer(cat: str, user: str, assistant: str) -> None:
    if cat == "algebra":
        a,sign,b,c = value(r"Loese (\d+)x ([+-]) (\d+) = (-?\d+)",user)
        x = int(value(r"^x = (-?\d+)\.",assistant)[0])
        assert int(a)*x+(1 if sign=="+" else -1)*int(b)==int(c)
    elif cat == "budget":
        price,quantity,discount = value(
            r"Ein Produkt kostet (\d+,\d{2}) EUR\. Ich kaufe (\d+) Stueck, auf die Gesamtsumme gibt es (\d+)%",user)
        unit=int(price.replace(",",""))
        after=unit*int(quantity)*(100-int(discount))
        if after%100:raise ValueError("Non-integer cents in budget")
        final=int(value(r"Mit \d+% Rabatt: (\d+,\d{2}) EUR",assistant)[0].replace(",",""))
        assert final==after//100
    elif cat == "fuel":
        burn,flight,reserve,taxi=value(
            r"konstant (\d+) kg/h\. Flugzeit (\d+) min, Reserve (\d+) min, Taxi-Fuel (\d+) kg",user)
        result=int(value(r"= (\d+) kg",assistant)[0])
        expected=int(burn)*(int(flight)+int(reserve))
        if expected%60:raise ValueError("Fuel result not integral")
        assert result==expected//60+int(taxi)
    elif cat == "timezones":
        hh,mm,a,dur,b=value(
            r"Abflug (\d{2}):(\d{2}) Ortszeit \(fest UTC([+-]\d+)\)\. Flugdauer (\d+) min\. Ziel hat fest UTC([+-]\d+)",user)
        minutes=int(hh)*60+int(mm)+int(dur)+60*(int(b)-int(a))
        target=minutes%(24*60);day=minutes//(24*60)
        expected_time=f"{target//60:02d}:{target%60:02d}"
        if f"Ankunft {expected_time}" not in assistant:
            raise ValueError("Incorrect target local clock time")
        suffix=f"naechster Tag (+{day})" if day>0 else (f"Vortag ({day})" if day<0 else "gleicher Tag")
        if suffix not in assistant: raise ValueError("Incorrect local day offset")
    elif cat in {"sql_join","sql_group"}:
        if not assistant.startswith("SELECT ") or not assistant.rstrip().endswith(";"):
            raise ValueError("SQL answer must be SELECT only")
        if re.search(r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|EXEC)\b",assistant,re.I):
            raise ValueError("Unsafe SQL label")
        if cat=="sql_group":
            minimum=int(value(r"mindestens (\d+) Zeilen",user)[0])
            floor=int(value(r"mit \w+ >= (\d+)",user)[0])
            if f"HAVING COUNT(*) >= {minimum}" not in assistant or f"WHERE " not in assistant or f">= {floor}" not in assistant:
                raise ValueError("Wrong SQL aggregation")
            if "GROUP BY" not in assistant or "ORDER BY" not in assistant:
                raise ValueError("SQL aggregation incomplete")
        elif "ohne " in user or "auch ohne " in user:
            if "LEFT JOIN" not in assistant:raise ValueError("Missing left join")
        elif "INNER JOIN" not in assistant:
            raise ValueError("Missing inner join")
    elif cat=="python_debug":
        name=value(r"(def solve_\d+\()",user)[0][4:-1]
        tree=ast.parse(assistant)
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!=name:
            raise ValueError("Invalid Python repair label")
        if any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Call)) and
               isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in
               {"exec","eval","open","compile","__import__"} for n in ast.walk(tree)):
            raise ValueError("Unsafe generated training code")
    elif cat=="tool_choice":
        answer=json.loads(assistant)
        if set(answer)!={"tool","requires_approval","reason"} or answer["requires_approval"] is not False:
            raise ValueError("Wrong tool authorization label")
        tool=answer["tool"]
        targets={"Liste alle":"inventory","Suche Python":"python","Pruefe lokale JSON":"formats",
                 "Suche auffaellige":"security","Finde eine":"search",
                 "Pruefe lokale Markdown":"markdown","Lies eine":"read"}
        matches=[v for k,v in targets.items() if k in user]
        if matches != [tool]:
            raise ValueError("Incorrect tool routing")
    elif cat=="analysis":
        data=[int(v) for v in value(r"Messreihe \[([0-9, ]+)\]",user)[0].split(",")]
        mean=float(value(r"Mittelwert = ([\d.]+)",assistant)[0])
        median=int(value(r"Median = (\d+)",assistant)[0])
        if mean != statistics.mean(data) or median != statistics.median(data):
            raise ValueError("Wrong statistics")
    elif cat=="physics":
        mass,acc=value(r"Masse (\d+) kg, konstante Beschleunigung (\d+) m/s",user)
        force=int(value(r"= (\d+) N",assistant)[0])
        assert force==int(mass)*int(acc)
    else:raise ValueError("Unknown topic")

def validate(root: Path) -> dict:
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version")!=1 or manifest.get("training_allowed") is not False:
        raise ValueError("Missing reviewed synthetic curriculum contract")
    if manifest.get("splits")!={key:{"rows":n} for key,n in EXPECTED.items()}:
        raise ValueError("Unexpected split size in manifest")
    if tuple(manifest.get("categories",[])) != CATEGORIES:
        raise ValueError("Unexpected curriculum categories")
    prompts=set()
    metrics={}
    for split in SPLITS:
        path=root/f"{split}.jsonl"
        if not path.is_file() or path.is_symlink() or path.stat().st_size>MAX_JSONL_BYTES:
            raise ValueError("Missing/oversized/linked split")
        entries=path.read_text(encoding="utf-8").splitlines()
        if len(entries)!=EXPECTED[split]:
            raise ValueError("Split count differs from manifest")
        counter=collections.Counter()
        for n,line in enumerate(entries,1):
            try:
                record=json.loads(line)
                if set(record)!={"messages"} or not isinstance(record["messages"],list) or len(record["messages"])!=3:
                    raise ValueError("Invalid row schema")
                messages=record["messages"]
                if [msg.get("role") for msg in messages]!=["system","user","assistant"]:
                    raise ValueError("Unexpected roles")
                if any(set(msg)!={"role","content"} or not isinstance(msg["content"],str)
                       or not msg["content"].strip() for msg in messages):
                    raise ValueError("Invalid message fields")
                if messages[0]["content"]!=SYSTEM:
                    raise ValueError("Unexpected system prompt")
                user,answer=messages[1]["content"],messages[2]["content"]
                if len(user)>900 or len(answer)>1000:
                    raise ValueError("Example exceeds conservative text budget")
                canonical=" ".join(user.casefold().split())
                if canonical in prompts: raise ValueError("Repeated question across dataset splits")
                prompts.add(canonical)
                if any(t in (user+"\n"+answer).lower() for t in ("sk_live_","github_pat_","api_key=","password=")):
                    raise ValueError("Potential sensitive material in synthetic corpus")
                cat=category(user)
                audit_answer(cat,user,answer)
                counter[cat]+=1
            except Exception as exc:
                raise ValueError(f"Invalid synthetic {split} example {n}: {type(exc).__name__}") from exc
        if set(counter)!=set(CATEGORIES) or len(set(counter.values()))!=1:
            raise ValueError("Unbalanced curriculum categories")
        metrics[split]={"rows":len(entries),"category_counts":dict(sorted(counter.items()))}
    return {"schema":1,"valid":True,"total_rows":sum(EXPECTED.values()),
            "split_stats":metrics,"unique_prompts":len(prompts),
            "answers_programmatically_checked":True,
            "tokenizer_validated":False,"training_performed":False,"quality_gain_proven":False}

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=Path(__file__).resolve().parents[1]/"datasets/neral_curriculum_v1")
    args=parser.parse_args(argv)
    print(json.dumps(validate(args.data),indent=2))

if __name__=="__main__":
    main()
