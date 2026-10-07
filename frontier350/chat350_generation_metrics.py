from __future__ import annotations
import statistics

def repeated_ngram_fraction(ids: list[int], n: int = 3) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    seen = set()
    repeated = 0
    for g in grams:
        if g in seen:
            repeated += 1
        seen.add(g)
    return repeated / len(grams)

def distinct_ngram_fraction(ids: list[int], n: int = 2) -> float:
    if len(ids) < n:
        return 1.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return len(set(grams)) / len(grams)

def summarize_generations(rows: list[dict], eos_id: int) -> dict:
    if not rows:
        raise ValueError("generation rows are empty")
    rep, d2, lengths, eos, utf8 = [], [], [], [], []
    exact = []
    for row in rows:
        ids = [int(x) for x in row.get("generated_ids", [])]
        hit = eos_id in ids
        content_ids = ids[:ids.index(eos_id)] if hit else ids
        text = str(row.get("text", ""))
        try:
            text.encode("utf8", "strict")
            utf8.append(1.0)
        except UnicodeError:
            utf8.append(0.0)
        rep.append(repeated_ngram_fraction(content_ids, 3))
        d2.append(distinct_ngram_fraction(content_ids, 2))
        lengths.append(len(content_ids))
        eos.append(float(hit))
        if "expected_exact" in row:
            exact.append(float(text.strip() == str(row["expected_exact"]).strip()))
    return {
        "cases": len(rows),
        "eos_fraction": statistics.fmean(eos),
        "mean_repeated_trigram_fraction": statistics.fmean(rep),
        "mean_distinct2_fraction": statistics.fmean(d2),
        "mean_generated_tokens_before_eos": statistics.fmean(lengths),
        "utf8_fraction": statistics.fmean(utf8),
        "exact_instruction_pct": (100.0 * statistics.fmean(exact)) if exact else None,
    }

def compare_health(baseline: dict, candidate: dict) -> dict:
    eos_ok = candidate["eos_fraction"] >= baseline["eos_fraction"]
    repetition_ok = candidate["mean_repeated_trigram_fraction"] <= baseline["mean_repeated_trigram_fraction"] + 1e-12
    utf8_ok = candidate["utf8_fraction"] >= 0.999
    exact_base = baseline.get("exact_instruction_pct")
    exact_cand = candidate.get("exact_instruction_pct")
    exact_ok = exact_cand is None or exact_base is None or exact_cand >= exact_base
    return {
        "eos_no_regression": eos_ok,
        "repetition_no_regression": repetition_ok,
        "utf8_ok": utf8_ok,
        "exact_instruction_no_regression": exact_ok,
        "health_no_regression": eos_ok and repetition_ok and utf8_ok and exact_ok,
    }
