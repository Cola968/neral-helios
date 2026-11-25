"""Owner-dispatched public web smoke questions. No benchmark data or model calls."""
import json
import time

from neral_v43_live_web import route_live_query, search_live_sources
from neral_web_search import WebSearch, is_explicit_news_query


QUESTIONS = {
    "aviation": "Welche aktuellen Nachrichten gibt es zum Flughafen BER?",
    "aviation_2": "Was gibt es Neues am Flughafen MUC?",
    "space": "Welche aktuellen Nachrichten gibt es zur NASA Artemis Mission?",
    "technology": "Welche aktuellen Nachrichten gibt es zu künstlicher Intelligenz?",
    "politics": "Welche aktuellen Nachrichten gibt es zum Deutschen Bundestag?",
    "sport": "Welche aktuellen Nachrichten gibt es zur Fußball Bundesliga?",
    "fact": "In welchem Jahr landete Apollo 11 auf dem Mond?",
    "software": "Welche Version von Python ist aktuell?",
    "weather": "Wie ist das Wetter morgen in Berlin?",
    "energy": "Welche aktuellen Nachrichten gibt es zu Photovoltaik?",
    "archaeology": "Aktuelle Nachrichten zur Archäologie",
    "database": "Welche Version von PostgreSQL ist aktuell?",
}


def main():
    failures = []
    for category, question in QUESTIONS.items():
        start = time.monotonic()
        routed = route_live_query(question)
        isolated = route_live_query(question, benchmark=True) is None
        result = search_live_sources(WebSearch(), routed) if routed else {"status":"not_routed","sources":[]}
        pages = [s for s in result["sources"] if s.get("page_read") is True]
        full = sum(s.get("article_chars",0)>=240 and not s.get("article_truncated") for s in pages)
        passed = bool(isolated and result["status"]=="ok" and pages and (full or not is_explicit_news_query(question)))
        print("GENERAL_LIVE", json.dumps(dict(category=category, status=result["status"], read_pages=len(pages),
              full_articles=full, seconds=round(time.monotonic()-start,3), passed=passed,
              urls=[s["url"] for s in pages]),ensure_ascii=True), flush=True)
        if not passed:
            failures.append(category)
    if failures:
        raise SystemExit("Live acceptance failed: "+", ".join(failures))


if __name__ == "__main__":
    main()
