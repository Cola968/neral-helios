from __future__ import annotations

"""Build a deterministic external DE<->EN audit suite without torch.

The input is a JSONL file with one pair per line and at least `de` and `en`
string fields. The generated suite is evaluation-only and is intentionally
source-independent so a non-Tatoeba benchmark such as FLORES can be used after
V6 without coupling the data builder to the model runtime.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

_WS = re.compile(r"\s+")
_TRIM = re.compile(r"^[\s'\"“”‘’.,!?;:()\[\]{}]+|[\s'\"“”‘’.,!?;:()\[\]{}]+$")


def clean(text: str) -> str:
    return _WS.sub(" ", text.replace("\x00", " ")).strip()


def key(text: str) -> str:
    return _TRIM.sub("", clean(text).lower())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_hash(de: str, en: str) -> str:
    return hashlib.sha256((de + "\0" + en).encode("utf-8")).hexdigest()


def exclusion_keys(paths: list[Path]) -> set[str]:
    blocked: set[str] = set()
    for path in paths:
        spec = json.loads(path.read_text(encoding="utf8"))
        for case in spec.get("cases") or []:
            for value in [case.get("preferred", ""), *(case.get("distractors") or [])]:
                k = key(str(value))
                if k:
                    blocked.add(k)
            prefix = str(case.get("prefix", ""))
            for quoted in re.findall(r'[\"“”\']([^\"“”\']+)[\"“”\']', prefix):
                k = key(quoted)
                if k:
                    blocked.add(k)
    return blocked


def read_pairs(path: Path, blocked: set[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at line {line_no}: {exc}") from exc
            de = clean(str(row.get("de", "")))
            en = clean(str(row.get("en", "")))
            if not de or not en:
                continue
            if len(de) > 500 or len(en) > 500:
                continue
            if key(de) in blocked or key(en) in blocked:
                continue
            digest = pair_hash(de, en)
            if digest in seen:
                continue
            seen.add(digest)
            rows.append((de, en))
    rows.sort(key=lambda p: pair_hash(*p))
    return rows


def build_cases(chosen: list[tuple[str, str]]) -> list[dict]:
    n = len(chosen)
    if n < 4:
        raise RuntimeError("need at least four admitted pairs")
    offsets = (37, 83, 149)
    cases: list[dict] = []
    for i, (de, en) in enumerate(chosen):
        de_distractors = [chosen[(i + off) % n][0] for off in offsets]
        en_distractors = [chosen[(i + off) % n][1] for off in offsets]
        cases.append(
            {
                "id": f"ext-en-de-{i+1:04d}",
                "direction": "en_to_de",
                "prefix": f'English: "{en}"\nGerman:',
                "preferred": " " + de,
                "distractors": [" " + x for x in de_distractors],
            }
        )
        cases.append(
            {
                "id": f"ext-de-en-{i+1:04d}",
                "direction": "de_to_en",
                "prefix": f'Deutsch: "{de}"\nEnglish:',
                "preferred": " " + en,
                "distractors": [" " + x for x in en_distractors],
            }
        )
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with `de` and `en` fields")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--source-license", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--exclude-cases", action="append", default=[])
    args = ap.parse_args()

    if args.pairs < 50:
        raise RuntimeError("external audit should use at least 50 source pairs")

    input_path = Path(args.input)
    exclude_paths = [Path(p) for p in args.exclude_cases]
    blocked = exclusion_keys(exclude_paths)
    pairs = read_pairs(input_path, blocked)
    if len(pairs) < args.pairs:
        raise RuntimeError(f"need {args.pairs} admitted pairs, found {len(pairs)}")

    chosen = pairs[: args.pairs]
    cases = build_cases(chosen)
    spec = {
        "schema": 1,
        "suite": "arctus-external-bilingual-audit-v1",
        "source": {
            "id": args.source_id,
            "license": args.source_license,
            "revision": args.source_revision,
            "input_sha256": sha256(input_path),
        },
        "pairs": len(chosen),
        "cases": cases,
        "directions": {"en_to_de": len(chosen), "de_to_en": len(chosen)},
        "training_overlap": False,
        "purpose": "independent evaluation-only bilingual confirmation; never training data",
        "excluded_case_files": [
            {"path": str(p), "sha256": sha256(p)} for p in exclude_paths
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf8")
    print(
        json.dumps(
            {
                "ok": True,
                "pairs": len(chosen),
                "cases": len(cases),
                "source": args.source_id,
                "input_sha256": spec["source"]["input_sha256"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
