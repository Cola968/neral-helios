"""Opt-in live release smoke test, separate from offline benchmarks."""
import json
import os
import time

CASES = (
    ("aviation", "Welche aktuellen Nachrichten gibt es zum Flughafen BER?"),
    ("aviation_2", "Was gibt es Neues am Flughafen MUC?"),
    ("space", "Welche aktuellen Nachrichten gibt es zur NASA Artemis Mission?"),
    ("technology", "Welche aktuellen Nachrichten gibt es zu künstlicher Intelligenz?"),
    ("politics", "Welche aktuellen Nachrichten gibt es zum Deutschen Bundestag?"),
    ("sport", "Welche aktuellen Nachrichten gibt es zur Fußball Bundesliga?"),
    ("science", "What is the latest news about the James Webb Space Telescope?"),
    ("economy", "Wie hoch ist der aktuelle EZB-Leitzins?"),
    ("fact", "In welchem Jahr landete Apollo 11 auf dem Mond?"),
    ("software", "Welche Version von Python ist aktuell?"),
    ("weather", "Wie ist das Wetter morgen in Berlin?"),
)


def evaluate_case(category, question, route, lookup):
    start = time.monotonic()
    benchmark_blocked = route(question, benchmark=True) is None
    query = route(question) if benchmark_blocked else None
    result = (lookup(query) if query else
              {"status": "not_routed" if benchmark_blocked else "isolation_failed",
               "sources": []})
    sources = result["sources"]
    read_pages = sum(source.get("page_read") is True for source in sources)
    # Extraction metadata is not a guarantee that the publisher serves the
    # complete article, so do not count this as proven publisher completeness.
    untruncated_extractions = sum(
        source.get("page_read") is True
        and source.get("article_chars", 0) >= 240
        and source.get("article_truncated") is False
        for source in sources
    )
    row = {
        "category": category,
        "status": result["status"],
        "read_pages": read_pages,
        "untruncated_extractions": untruncated_extractions,
        "seconds": round(time.monotonic() - start, 3),
        "benchmark_blocked": benchmark_blocked,
        "passed": benchmark_blocked and result["status"] == "ok" and read_pages > 0,
    }
    counts = result.get("stage_counts")
    if isinstance(counts, dict):
        from neral_v43_live_web import LIVE_STAGE_COUNT_KEYS
        row["stage_counts"] = {
            key: counts[key] for key in LIVE_STAGE_COUNT_KEYS
            if type(counts.get(key)) is int and counts[key] >= 0
        }
    return row


def main():
    from neral_v43_live_web import route_live_query, search_live_sources
    from neral_web_search import WebSearch

    rows = []
    for category, question in CASES:
        row = evaluate_case(category, question, route_live_query,
                            lambda query: search_live_sources(WebSearch(), query, include_diagnostics=True))
        print("GENERAL_LIVE " + json.dumps(row), flush=True)
        rows.append(row)
    failures = [row["category"] for row in rows if not row["passed"]]
    summary = f"V4.3 live evidence: {len(rows) - len(failures)}/{len(rows)} passed."
    if failures:
        summary += " Failed: " + ", ".join(failures) + "."
    print(summary, flush=True)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(summary + "\n")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
