"""NERAL v5: 120 new quantitative word problems, a read-only Windows GPU A/B.

Fresh deterministic generator with distinct phrasing and structures. NO cases
from NERAL v5 train/validation/test generators. This is separately authored
synthetic transfer testing, not an independently sourced external benchmark.
Only scalar aggregate results can be exported; responses stay local.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import gc
import hashlib
import json
from pathlib import Path
import random
import re

from scripts.neral_v5_style_eval import (
    BASE_REVISION, SYSTEM, grade,
)
from scripts.neral_v3_rank_ab_eval import exact_mcnemar_p

CATEGORIES = ("affine", "fractions", "money", "fuel", "timezone", "quantities")
COUNT = 20


def _round_cents(value: Fraction) -> int:
    if value < 0:
        raise ValueError("Negative money")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def make_case(category: str, index: int) -> dict:
    if category not in CATEGORIES or type(index) is not int or not 0 <= index < COUNT:
        raise ValueError("Invalid novel math case coordinates")
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f"NERAL-NEW-MATH-120-20261107|{category}|{index}".encode()).digest()[:16], "big"))
    if category == "affine":
        a = 4 + (index % 11)
        solution = rng.randint(-32, 35)
        b = rng.randint(-38, 38)
        rhs = a * solution + b
        prompt = (
            f"Bei einer linearen Beziehung ergibt das {a}-Fache einer "
            f"unbekannten Zahl zusammen mit der Konstanten ({b}) den Wert "
            f"{rhs}. Welche Zahl ist gesucht? Gib nur x = Zahl zurück."
        )
        answer = f"x = {solution}"
    elif category == "fractions":
        left = Fraction(rng.randint(-23, 28), rng.randint(3, 18))
        right = Fraction(rng.randint(1, 27), rng.randint(3, 21))
        operation = ("addiert", "abgezogen", "multipliziert", "dividiert")[index % 4]
        value = (
            left + right if index % 4 == 0 else
            left - right if index % 4 == 1 else
            left * right if index % 4 == 2 else
            left / right
        )
        prompt = (
            f"Zwei rationale Werte sind {left} und {right}. "
            f"Der erste wird mit dem zweiten {operation}. "
            f"Wie lautet das vollständig gekürzte Resultat? "
            f"Antworte nur als gekürzter Bruch oder ganze Zahl."
        )
        answer = str(value)
    elif category == "money":
        cents = 725 + index * 173 + rng.randint(0, 38)
        surcharge = rng.randint(7, 39)
        discount = rng.randint(3, 26)
        result_cents = _round_cents(Fraction(
            cents * (100 + surcharge) * (100 - discount), 10_000))
        prompt = (
            f"Eine Dienstleistung kostet zunächst {cents//100},{cents%100:02d} EUR. "
            f"Zuerst wird sie um {surcharge}% teurer. Vom neuen Preis werden "
            f"danach {discount}% abgezogen. Welcher Endbetrag ergibt sich "
            f"nach kaufmännischer Rundung? Nur Betrag mit zwei Kommastellen und EUR."
        )
        answer = f"{result_cents//100},{result_cents%100:02d} EUR"
    elif category == "fuel":
        rate = 75 * rng.randint(5, 24)
        hours = rng.randint(1, 8)
        minutes = rng.choice((15, 30, 45))
        reserve = rng.choice((15, 30, 60, 90))
        startup = rng.randint(15, 185)
        total = Fraction(rate * (hours * 60 + minutes + reserve), 60) + startup
        prompt = (
            f"Eine Maschine verbraucht {rate} kg Treibstoff je Stunde. "
            f"Die Betriebszeit beträgt {hours} h {minutes} min; "
            f"zusätzlich sind {reserve} min Sicherheitsreserve und "
            f"ein einmaliger Startverbrauch von {startup} kg einzuplanen. "
            f"Welche Gesamtmasse ist erforderlich? Gib den exakten Wert in kg."
        )
        answer = f"{total} kg"
    elif category == "timezone":
        local_hour = rng.randint(0, 23)
        minute = rng.choice((0, 15, 30, 45))
        src = rng.randint(-9, 11)
        dst = rng.randint(-10, 10)
        journey = 35 + index * 67
        absolute = local_hour * 60 + minute + journey + (dst-src)*60
        day, local_minutes = divmod(absolute, 1440)
        prompt = (
            f"Für eine Reise gelten ausdrücklich feste Zeitzonen ohne Sommerzeit. "
            f"Die Abfahrt ist um {local_hour:02d}:{minute:02d} Uhr in UTC{src:+d}; "
            f"die Fahrt dauert {journey} Minuten. Das Ziel liegt in UTC{dst:+d}. "
            f"Welche lokale Zielzeit und relative Tageszahl ergeben sich? "
            f"Schreibe exakt HH:MM (Tag +/-N)."
        )
        answer = f"{local_minutes//60:02d}:{local_minutes%60:02d} (Tag {day:+d})"
    else:
        count_per_box = rng.randint(8, 35)
        boxes = 4 + index % 15
        delivery = rng.randint(5, 51)
        broken = rng.randint(1, 24)
        total = boxes * count_per_box + delivery - broken
        prompt = (
            f"Im Lager stehen {boxes} Kartons mit je {count_per_box} Artikeln. "
            f"Eine neue Lieferung bringt weitere {delivery} Artikel. "
            f"{broken} Artikel werden aussortiert. "
            f"Wie viele brauchbare Artikel sind jetzt vorhanden? "
            f"Antworte nur mit ganzer Zahl und dem Wort Stück."
        )
        answer = f"{total} Stück"
    return {"id": f"{category}-{index:02d}", "category": category,
            "prompt": prompt, "answer": answer}


CASES = tuple(make_case(cat, i) for cat in CATEGORIES for i in range(COUNT))
_IDS = {x["id"] for x in CASES}


def case_hash() -> str:
    raw = json.dumps(CASES, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def verify_novel_prompts(source: Path, cases=CASES) -> bool:
    from scripts.neral_v5_transfer_curriculum import verify
    if not verify(Path(source))["valid"]:
        raise ValueError("Original v5 dataset integrity failed")
    if len(cases) != 120 or len({c["id"] for c in cases}) != 120:
        raise ValueError("Expected 120 distinct novel math cases")
    own = {" ".join(c["prompt"].casefold().split()) for c in cases}
    if len(own) != 120:
        raise ValueError("Novel math prompts must be unique")
    original = set()
    for split in ("train", "val", "test"):
        for line in (Path(source)/(split+".jsonl")).read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            original.add(" ".join(row["prompt"][1]["content"].casefold().split()))
    if original & own:
        raise ValueError("Novel math evaluation prompt overlaps v5 train/val/test split")
    if any(grade(c, c["answer"]) != (True, True) for c in cases):
        raise ValueError("Invalid novel math oracle label")
    return True


def _scores(rows: list[dict]) -> dict:
    if (not isinstance(rows, list) or len(rows) != 120 or
            any(not isinstance(row, dict) or set(row) !=
                {"id", "correct", "format_valid"} or
                row["id"] not in _IDS or
                type(row["correct"]) is not bool or
                type(row["format_valid"]) is not bool for row in rows)):
        raise ValueError("Invalid 120-case scores")
    mapping = {row["id"]: row for row in rows}
    if set(mapping) != _IDS:
        raise ValueError("Duplicated or missing novel math case IDs")
    return mapping


def report(base: list[dict], neral: list[dict], *,
           adapter_sha: str, base_revision: str) -> dict:
    if (not isinstance(adapter_sha, str) or
            not re.fullmatch(r"[0-9a-f]{64}", adapter_sha) or
            base_revision != BASE_REVISION):
        raise ValueError("Unverified model provenance")
    b, n = _scores(base), _scores(neral)
    gain = sum(not b[k]["correct"] and n[k]["correct"] for k in b)
    loss = sum(b[k]["correct"] and not n[k]["correct"] for k in b)
    per_category = {}
    for cat in CATEGORIES:
        keys = [case["id"] for case in CASES if case["category"] == cat]
        per_category[cat] = {
            "total": len(keys),
            "base_correct": sum(b[k]["correct"] for k in keys),
            "neral_correct": sum(n[k]["correct"] for k in keys),
            "base_format": sum(b[k]["format_valid"] for k in keys),
            "neral_format": sum(n[k]["format_valid"] for k in keys),
        }
    return {
        "schema": 1, "status": "NERAL_V5_NOVEL_SYNTHETIC_MATH_120",
        "total": 120,
        "base": {"correct": sum(x["correct"] for x in b.values()),
                 "format_valid": sum(x["format_valid"] for x in b.values())},
        "neral_v5": {"correct": sum(x["correct"] for x in n.values()),
                     "format_valid": sum(x["format_valid"] for x in n.values())},
        "per_category": per_category,
        "paired": {"improvements": gain, "regressions": loss,
                   "net_gain": gain - loss, "discordant": gain + loss,
                   "exact_mcnemar_p": exact_mcnemar_p(gain, loss)},
        "base_revision": base_revision,
        "adapter_sha256": adapter_sha, "case_hash": case_hash(),
        "trained_adapter_evaluated": True,
        "independent_external_benchmark": False,
        "general_improvement_proven": False,
        "release_validated": False, "weights_promoted": False,
        "limitations": "Fresh, structurally different synthetic math families independent of the v5 generator; same research domain, not external or blinded. Strict formats and arithmetic correctness.",
    }


def combine(root: Path, *, adapter_sha: str, base_revision: str) -> dict:
    scores = {}
    for variant in ("base", "neral_v5"):
        folder = Path(root)/variant
        path = folder/"local.json"
        if (folder.is_symlink() or path.is_symlink() or
                not path.is_file() or path.stat().st_size > 150_000):
            raise ValueError("Missing or unsafe local model score")
        data = json.loads(path.read_text(encoding="utf8"))
        if (not isinstance(data, dict) or set(data) !=
                {"variant", "adapter_sha256", "base_revision", "case_hash", "scores"} or
                data["variant"] != variant or
                data["adapter_sha256"] != adapter_sha or
                data["base_revision"] != base_revision or
                data["case_hash"] != case_hash()):
            raise ValueError("Mismatched saved A/B provenance")
        raw = data["scores"]
        if (not isinstance(raw, list) or len(raw) != 120 or
                any(not isinstance(x, dict) or set(x) !=
                    {"id", "correct", "format_valid", "answer_sha256", "local_raw"} or
                    not isinstance(x["answer_sha256"], str) or
                    not re.fullmatch("[0-9a-f]{64}", x["answer_sha256"]) or
                    not isinstance(x["local_raw"], str) or
                    len(x["local_raw"]) > 250 for x in raw)):
            raise ValueError("Unsafe raw response inventory")
        scores[variant] = [
            {"id": x["id"], "correct": x["correct"],
             "format_valid": x["format_valid"]} for x in raw
        ]
    return report(scores["base"], scores["neral_v5"],
                  adapter_sha=adapter_sha, base_revision=base_revision)


def evaluate(model, tokenizer) -> list[dict]:
    import torch
    from qwen_local_runtime import CHAT_TEMPLATE
    model.eval()
    scores = []
    for case in CASES:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": case["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, chat_template=CHAT_TEMPLATE,
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(
                inputs, attention_mask=torch.ones_like(inputs),
                do_sample=False, max_new_tokens=96,
                pad_token_id=tokenizer.pad_token_id)
        raw = tokenizer.decode(generated[0,inputs.shape[-1]:],skip_special_tokens=True)
        correct, valid = grade(case, raw)
        scores.append({
            "id": case["id"], "correct": bool(correct),
            "format_valid": bool(valid),
            "answer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "local_raw": raw[:250],
        })
        del inputs, generated
    return scores


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("base", "adapter", "trainer-root", "source-root", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--variant", choices=("base", "neral_v5"))
    p.add_argument("--combine", action="store_true")
    args = p.parse_args(argv)
    if args.combine == bool(args.variant):
        p.error("Select exactly one --variant or --combine")
    if args.out.is_symlink() or args.out.exists() or not args.out.parent.is_dir():
        p.error("Output path must be new")
    verify_novel_prompts(args.source_root)
    import sys
    root = args.trainer_root.resolve(strict=True)
    if args.trainer_root.is_symlink() or not root.is_dir():
        p.error("Pinned trainer missing")
    sys.path.insert(0, str(root))
    from model_catalog import validate_hf_snapshot
    from neral_adapter_candidate import validate_adapter
    snapshot = validate_hf_snapshot(args.base, verify_hashes=True)
    if snapshot["revision"] != BASE_REVISION:
        p.error("Pinned Qwen revision differs")
    manifest = validate_adapter(args.adapter, snapshot)
    if manifest["release_validated"] is not False:
        p.error("Only nonpromoted research adapter is allowed")
    sha = manifest["files"]["adapter_model.safetensors"]
    if args.combine:
        result = combine(args.out.parent, adapter_sha=sha,
                         base_revision=BASE_REVISION)
        args.out.mkdir(exist_ok=False)
        (args.out/"summary.json").write_text(json.dumps(
            result, sort_keys=True, indent=2)+"\n", encoding="utf8")
        print(json.dumps(result, sort_keys=True))
        return
    from peft import PeftModel
    from transformers import Qwen2TokenizerFast
    from scripts.neral_math_eval import _load_model
    from scripts.neral_v3_eval_isolation import prepare_cuda, clean_cuda
    import torch
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        str(args.base), local_files_only=True, trust_remote_code=False, token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepare_cuda(torch)
    model = None
    args.out.mkdir(exist_ok=False)
    try:
        model = _load_model(args.base)
        if args.variant == "neral_v5":
            model = PeftModel.from_pretrained(
                model, str(args.adapter), is_trainable=False, local_files_only=True)
        scores = evaluate(model, tokenizer)
        (args.out/"local.json").write_text(json.dumps({
            "variant": args.variant, "adapter_sha256": sha,
            "base_revision": BASE_REVISION, "case_hash": case_hash(),
            "scores": scores
        }, indent=2)+"\n", encoding="utf8")
        print(f"NERAL_V5_NOVEL_MATH_{args.variant}: 120 cases; raw text stays local")
    finally:
        if model is not None:
            del model
        gc.collect()
        clean_cuda(torch)


if __name__ == "__main__":
    main()
