"""Arctus350 closed-world, bilingual *data-only* instruction curriculum v1.

Exercises grounded fact retrieval, abstaining when facts are missing, and
comparing two given measurements. These synthetic records are NOT facts about
the world, native model completions, or independent benchmark evidence.

- No network, GPU, external teachers, downloads, model weights or training.
- Every row is reproducible and independently verified from its ID.
- All six language/task variants of an invented entity stay in ONE split.
- This does not prove non-contamination with preexisting private datasets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .arctus350_exact_math_curriculum import SOURCE_LICENSE

SOURCE = "neral-generated-closed-world-grounding-v1"
SCHEMA = "arctus350-closed-world-grounding-curriculum-v1"
TOTAL_ENTITIES = 360
_SPLITS = ("train", "val", "test")
_ROW_ID = re.compile(r"cw-(\d{4})-(de|en)-(temperature|mass_known|mass_missing|compare)\Z")


def _attributes(n: int) -> dict:
    """Re-derive all supposed 'facts' from one numbered fictional record."""
    if type(n) is not int or not 0 <= n < TOTAL_ENTITIES:
        raise ValueError("invalid synthetic entity number")
    digest = hashlib.sha256(f"arctus350-grounding-v1:{n}".encode("ascii")).digest()
    return {
        "object": f"K-{n:04d}",
        "temperature": 205 + digest[0] % 180,
        "pressure": 950 + digest[1] % 80,
        "mass": 7 + digest[2] % 78,
        "sensor_a": 1 + digest[3] % 70,
        "sensor_b": 1 + digest[4] % 70,
        "has_mass": n % 2 == 0,
    }


def _task(n: int, lang: str, family: str) -> tuple[str, str, str]:
    if lang not in ("de", "en"):
        raise ValueError("expected DE/EN language")
    attrs = _attributes(n)
    oid, t, p, m = (attrs[k] for k in ("object", "temperature", "pressure", "mass"))
    if family == "temperature":
        if lang == "de":
            prompt = (f"Fiktiver Datensatz: Objekt {oid}; Temperatur = {t} K; Druck = {p} kPa. "
                      "Welche Temperatur steht im Datensatz? Antworte nur mit Wert und Einheit.")
        else:
            prompt = (f"Fictional record: Object {oid}; Temperature = {t} K; Pressure = {p} kPa. "
                      "What temperature is listed? Reply only with value and unit.")
        return "grounding", prompt, f"{t} K"

    if family == "mass":
        extra = (f"; Masse = {m} kg" if lang == "de" else f"; Mass = {m} kg") if attrs["has_mass"] else ""
        if lang == "de":
            prompt = (f"Fiktiver Datensatz: Objekt {oid}; Temperatur = {t} K{extra}. "
                      "Welche Masse steht im Datensatz? Falls keine Masse angegeben ist, "
                      "schreibe exakt: nicht angegeben.")
            result = f"{m} kg" if attrs["has_mass"] else "nicht angegeben"
        else:
            prompt = (f"Fictional record: Object {oid}; Temperature = {t} K{extra}. "
                      "What mass is listed? If no mass is given, reply exactly: not provided.")
            result = f"{m} kg" if attrs["has_mass"] else "not provided"
        return ("mass_known" if attrs["has_mass"] else "mass_missing"), prompt, result

    if family == "compare":
        a,b=attrs["sensor_a"],attrs["sensor_b"]
        if lang == "de":
            prompt = (f"Fiktiver Datensatz: Objekt {oid}; Sensor A = {a}; Sensor B = {b}. "
                      "Welcher Sensor hat den größeren Messwert? Antworte nur A, B oder gleich.")
            answer = "A" if a > b else "B" if b > a else "gleich"
        else:
            prompt = (f"Fictional record: Object {oid}; Sensor A = {a}; Sensor B = {b}. "
                      "Which sensor has the larger reading? Reply only A, B or equal.")
            answer = "A" if a > b else "B" if b > a else "equal"
        return "compare", prompt, answer

    raise ValueError("unsupported grounding task family")


def _make_row(n, lang, family):
    kind, prompt, answer = _task(n,lang,family)
    return {
        "id": f"cw-{n:04d}-{lang}-{kind}",
        "language": lang,
        "domain": "grounding",
        "task_type": kind,
        "source": SOURCE,
        "license": SOURCE_LICENSE,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def entity_id(row: dict) -> str:
    if not isinstance(row,dict):
        raise ValueError("invalid row")
    match = _ROW_ID.fullmatch(row.get("id","")) if isinstance(row.get("id"),str) else None
    if match is None:
        raise ValueError("invalid closed-world identity")
    n=int(match.group(1))
    _attributes(n)
    return f"K-{n:04d}"


def verify_row(row: dict) -> bool:
    """Reject *any* label, prompt, ID or provenance drift, not just math errors."""
    if not isinstance(row,dict) or set(row)!={
        "id","language","domain","task_type","source","license","messages",
    }:
        return False
    match=_ROW_ID.fullmatch(row["id"]) if isinstance(row["id"],str) else None
    if match is None:
        return False
    n,lang,kind=int(match.group(1)),match.group(2),match.group(3)
    if n>=TOTAL_ENTITIES:
        return False
    family="mass" if kind.startswith("mass_") else kind
    try:
        return row==_make_row(n,lang,family)
    except (ValueError,TypeError):
        return False


def _partition(n: int) -> str:
    digest=hashlib.sha256(f"arctus350-grounding-split-v1:{n}".encode("ascii")).digest()
    bucket=int.from_bytes(digest[:4],"big") % 10
    return "train" if bucket<8 else "val" if bucket==8 else "test"


def make_splits() -> dict[str,list[dict]]:
    """Reproducible, entity-disjoint across both languages and all task types."""
    out={name:[] for name in _SPLITS}
    for n in range(TOTAL_ENTITIES):
        target=_partition(n)
        for lang in ("de","en"):
            for family in ("temperature","mass","compare"):
                row=_make_row(n,lang,family)
                if not verify_row(row):
                    raise AssertionError("independent reference regeneration failure")
                out[target].append(row)
    owners={}
    ids=set()
    for split,rows in out.items():
        if len(rows)<48:
            raise ValueError("insufficient split coverage")
        seen=Counter((r["language"],r["task_type"]) for r in rows)
        if any(seen[lang,kind]==0 for lang in ("de","en")
               for kind in ("temperature","mass_known","mass_missing","compare")):
            raise ValueError("missing task/language split coverage")
        for row in rows:
            if row["id"] in ids:
                raise ValueError("duplicate synthetic instruction ID")
            ids.add(row["id"])
            eid=entity_id(row)
            previous=owners.setdefault(eid,split)
            if previous!=split:
                raise ValueError("entity-disjoint heldout boundary breached")
    if len(owners)!=TOTAL_ENTITIES or len(ids)!=TOTAL_ENTITIES*6:
        raise AssertionError("generated corpus totals do not match")
    return out


def _sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write_local_dataset(dest) -> dict:
    """Only create a NEW local directory; emits aggregates, no private data."""
    dest=Path(dest)
    if dest.exists():
        raise FileExistsError("never overwrite local training/evaluation data")
    splits=make_splits()
    dest.mkdir(parents=True,exist_ok=False)
    for split,rows in splits.items():
        with (dest/(split+".jsonl")).open("x",encoding="utf8") as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    manifest,readback=validate_dataset_root(dest)
    validate_prompt_holdouts(readback)
    # Compare the reloaded rows after schema normalization, avoiding optional metadata.
    for split in _SPLITS:
        for row in splits[split]:
            if not verify_row(row):
                raise AssertionError("reference integrity drift")
    report={
        "schema":SCHEMA,"source":SOURCE,"status":"prepared_local_only",
        "counts":{s:len(splits[s]) for s in _SPLITS},
        "language_counts":{s:dict(sorted(Counter(r["language"] for r in splits[s]).items())) for s in _SPLITS},
        "task_counts":{s:dict(sorted(Counter(r["task_type"] for r in splits[s]).items())) for s in _SPLITS},
        "total_entities":TOTAL_ENTITIES,"total_rows":TOTAL_ENTITIES*6,
        "manifest_sha256":manifest["manifest_sha256"],
        "source_file_sha256":{s:_sha_file(dest/(s+".jsonl")) for s in _SPLITS},
        "all_references_verified":True,"entity_holdouts_disjoint":True,
        "training_started":False,"release_allowed":False,"benchmark_claimed":False,
        "native_intelligence_improved":False,
        "limitations":"Synthetic DE/EN closed-world microtasks; not a general reasoning benchmark. "
                      "Unseen entities are split-disjoint, but overlapping numeric values and "
                      "unknown external corpora cannot be ruled out.",
    }
    (dest/"metadata_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    summary=write_local_dataset(args.out)
    print(json.dumps(summary,sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
