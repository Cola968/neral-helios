"""NERAL Qwen3.5 structured-constraint SFT curriculum (CPU only).

Deterministic synthetic data, not a claim of semantically independent evaluation.
Do NOT include holdout records, benchmark prompts or model-authored answers.
ChatML is NOT assembled here: train with the actual tokenizer chat template.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

from neral_formal_constraints import PREFIX, try_solve_formal_constraints
from neral_turn0_extractor import EXTRACT_PROMPT, parse_extracted_constraints

TRAIN_SEED = 2026111201
REPAIR_PROMPT = (
    "\nONE SOLVER REPAIR TURN. Previous model schema rejected or UNSAT. "
    "Re-read the ORIGINAL question and every requirement; keep evidence, "
    "domains and all clauses. Do not invent source facts or drop conditions. "
    "Return ONE complete corrected strict JSON schema, no explanation, "
    "no Markdown, no code fences. Begin immediately with { and end with }."
)
BOOLEAN_NAMES = ("A", "B", "C", "D", "E", "F", "G")
ORDER_NAMES = ("Ada", "Bela", "Cleo", "Dara", "Egon", "Fina", "Gero")

def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def evidence(question_parts: tuple[str, str, str]) -> list[str]:
    return list(question_parts)

def boolean_task(rng: random.Random) -> tuple[str, dict]:
    n = rng.randrange(4, 7)
    names = list(BOOLEAN_NAMES[:n])
    parts: list[tuple[str, dict]] = []
    seen = set()
    for _ in range(rng.choice((2, 3, 4))):
        a, b = rng.sample(names, 2)
        kind = rng.choice(("or", "and", "xor", "implies", "not", "exactly"))
        if kind == "or":
            statement, expr = f"{a} OR {b}", {"op":"or","args":[a,b]}
        elif kind == "and":
            statement, expr = f"{a} AND {b}", {"op":"and","args":[a,b]}
        elif kind == "xor":
            statement, expr = f"{a} XOR {b} is True", {"op":"eq","args":[{"op":"xor","args":[a,b]},True]}
        elif kind == "implies":
            statement, expr = f"{a} implies {b}", {"op":"implies","args":[a,b]}
        elif kind == "not":
            statement, expr = f"{a} is False", {"op":"not","args":[a]}
        else:
            members = rng.sample(names, rng.choice((3, 4)))
            k = rng.randrange(1, len(members))
            statement, expr = (
                f"Exactly {k} of {', '.join(members)} are True",
                {"op":"exactly","count":k,"args":members},
            )
        if statement in seen:
            continue
        seen.add(statement)
        parts.append((statement, expr))
    head = f"Boolean variables {', '.join(names)} are each True or False."
    middle = "Requirements: " + "; ".join(item[0] for item in parts) + "."
    tail = "How many assignments satisfy all requirements?"
    question = " ".join((head, middle, tail))
    envelope = {
        "kind":"formal","complete":True,
        "schema":{
            "variables":[{"name":n,"domain":[False,True]} for n in names],
            "rules":[],"bool_rules":[p[1] for p in parts],"query":"count",
        },
        "evidence":evidence((head,middle,tail)),
    }
    return question, envelope

def ordering_task(rng: random.Random) -> tuple[str, dict]:
    n = rng.randrange(4, 7)
    names = rng.sample(ORDER_NAMES, n)
    count = rng.choice((2, 3))
    pairs = rng.sample([(a,b) for a in names for b in names if a != b], count)
    statements = [f"{a} must be before {b}" for a,b in pairs]
    head = f"{', '.join(names)} are all distinct and each appears once in a complete order."
    middle = "Conditions: " + "; ".join(statements) + "."
    tail = "How many complete orders satisfy all conditions?"
    question = " ".join((head,middle,tail))
    envelope = {
        "kind":"formal","complete":True,
        "schema":{"ordering":names,
                  "rules":[{"left":a,"op":"<","right":b} for a,b in pairs],
                  "query":"count"},
        "evidence":evidence((head,middle,tail)),
    }
    return question,envelope

def validated(question: str, good: dict) -> int | None:
    """Both the formal solver and the full source-grounding parser must agree."""
    schema=good["schema"]
    if "ordering" not in schema:
        compiled = copy.deepcopy(schema)
        compiled["variables"]={v["name"]:v["domain"] for v in schema["variables"]}
    else:
        compiled = schema
    try:
        direct=try_solve_formal_constraints(PREFIX+compact(compiled))
        rejected=[]
        parsed=parse_extracted_constraints(question,compact(good),
                                           rejection_reason_sink=rejected)
    except (ValueError,TypeError):
        return None
    if type(direct) is not int or direct != parsed or rejected != ["accepted"]:
        return None
    if direct <= 0:
        return None  # Train only satisfiable source tasks.
    return direct

def bad_proposal(good: dict, *, kind: str) -> tuple[dict,dict]:
    bad=copy.deepcopy(good)
    schema=bad["schema"]
    if kind == "solver_type_error":
        if "ordering" in schema:
            raise ValueError("Boolean type-error repair needs Boolean variables")
        first=schema["variables"][0]["name"]
        schema["rules"].append({"left":first,"op":"==","value":1})
        feedback={"rejection_reason":"solver_type_error",
                  "solver_rejection_detail":"comparison_types"}
    elif kind == "solver_unsat_fallback":
        if "ordering" in schema:
            a,b=schema["ordering"][:2]
            schema["rules"] += [
                {"left":a,"op":"<","right":b},
                {"left":b,"op":"<","right":a},
            ]
        else:
            first=schema["variables"][0]["name"]
            schema["bool_rules"].append(
                {"op":"and","args":[first,{"op":"not","args":[first]}]}
            )
        feedback={"rejection_reason":"solver_unsat_fallback",
                  "solver_solution_count_bucket":"zero_unsat"}
    else:
        raise ValueError("Unknown synthetic diagnostic")
    return bad,feedback

def make_row(turns: list[dict], question: str, family: str, mode: str,
             count: int) -> dict:
    return {
        "messages":turns,
        "meta":{"split":"train","source":"synthetic_oracle",
                "family":family,"mode":mode,
                "question_sha256":digest(question),
                "answer_count":count},
    }

def generate(*, pairs:int=360, seed:int=TRAIN_SEED,
             exclusion_questions:frozenset[str]=frozenset()) -> list[dict]:
    if not 250 <= pairs <= 500:
        raise ValueError("Requested paired data must produce 500-1000 rows")
    rng=random.Random(seed)
    rows=[]
    seen=set(exclusion_questions)
    attempts=0
    while len(rows) < pairs*2:
        attempts+=1
        if attempts > pairs*250:
            raise RuntimeError("Cannot fill validated curriculum")
        family="boolean" if len(rows)//2 % 4 != 3 else "ordering"
        question, good=(boolean_task(rng) if family=="boolean"
                        else ordering_task(rng))
        signature=digest(question)
        if signature in seen:
            continue
        count=validated(question,good)
        if count is None:
            continue
        kind=("solver_type_error" if family=="boolean" and
              (len(rows)//2)%2==0 else "solver_unsat_fallback")
        bad,feedback=bad_proposal(good,kind=kind)
        # An invalid first proposal MUST be rejected (or give UNSAT) by
        # the actual strict solver, never by a model-predicted label.
        bad_schema=copy.deepcopy(bad["schema"])
        if type(bad_schema.get("variables")) is list:
            bad_schema["variables"]={v["name"]:v["domain"]
                                     for v in bad_schema["variables"]}
        try:
            bad_count=try_solve_formal_constraints(PREFIX+compact(bad_schema))
        except ValueError:
            bad_count=None
        if kind=="solver_type_error" and bad_count is not None:
            continue
        if kind=="solver_unsat_fallback" and bad_count != 0:
            continue
        target=compact(good)
        rows.append(make_row([
            {"role":"system","content":EXTRACT_PROMPT},
            {"role":"user","content":question},
            {"role":"assistant","content":target},
        ],question,family,"turn0",count))
        payload={"original_question":question,
                 "previous_model_schema":compact(bad),
                 "solver_feedback":feedback}
        rows.append(make_row([
            {"role":"system","content":EXTRACT_PROMPT+REPAIR_PROMPT},
            {"role":"user","content":compact(payload)},
            {"role":"assistant","content":target},
        ],question,family,"turn1",count))
        seen.add(signature)
    return rows

def load_holdout(path:Path) -> tuple[frozenset[str],set[str]]:
    seen=set()
    shapes=set()
    if not path.is_file():
        raise FileNotFoundError("Holdout manifest/data file required")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        if set(row) < {"question","schema"}:
            raise ValueError("Invalid holdout row")
        question=row["question"]
        if type(question) is not str:
            raise ValueError("Invalid holdout question")
        key=digest(question)
        if key in seen:
            raise ValueError("Duplicate holdout item")
        seen.add(key)
        shapes.add(digest(compact(row["schema"])))
    if not 20 <= len(seen) <= 30:
        raise ValueError("Holdout must contain 20-30 independent questions")
    return frozenset(seen),shapes

def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--holdout",required=True,type=Path)
    parser.add_argument("--pairs",type=int,default=360)
    parser.add_argument("--seed",type=int,default=TRAIN_SEED)
    args=parser.parse_args()
    excluded,holdout_shapes=load_holdout(args.holdout)
    rows=generate(pairs=args.pairs,seed=args.seed,
                  exclusion_questions=excluded)
    if len(rows)!=args.pairs*2:
        raise AssertionError("Unexpected training row count")
    # Guard against same exact canonical formal schema, not just exact text.
    for row in rows:
        answer=json.loads(row["messages"][-1]["content"])
        if digest(compact(answer["schema"])) in holdout_shapes:
            raise RuntimeError("Holdout formal-schema leakage")
        if row["meta"]["question_sha256"] in excluded:
            raise RuntimeError("Holdout prompt leakage")
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Refuse to overwrite a prior dataset")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8",newline="\n") as f:
        for row in rows:
            f.write(compact(row)+"\n")
    manifest={"seed":args.seed,"rows":len(rows),"pairs":args.pairs,
              "turn0":args.pairs,"turn1":args.pairs,
              "holdout_count":len(excluded),
              "train_file_sha256":hashlib.sha256(args.out.read_bytes()).hexdigest(),
              "split":"train_only_no_holdout"}
    (args.out.parent/"train_manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(compact(manifest))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
