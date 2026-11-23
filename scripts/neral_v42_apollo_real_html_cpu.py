"""Read the complete public Apollo 11 Wikipedia HTML on CPU, no GPU inference.

This is an opt-in live-public-source preflight for the single diagnostic PR.
No model is loaded, no private/sealed test cases are used or captured.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from neral_page_reader import MainText
from neral_v42_quality import unique_read_year_source, year_source_pipeline_code

ARTICLE = "https://de.wikipedia.org/wiki/Apollo_11"
REST_HTML = "https://de.wikipedia.org/w/rest.php/v1/page/Apollo_11/html"
QUERY = "Apollo 11 erste bemannte Mondlandung Jahr NASA"
QUESTION = (
    "In welchem Jahr landete Apollo 11 erstmals Menschen auf dem Mond? "
    "Nenne die Jahreszahl und belege sie mit einem wörtlichen Zitat "
    "aus einer tatsächlich gelesenen Quelle."
)
MAX_HTML_BYTES = 8 * 1024 * 1024
YEAR = re.compile(r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?!\d)")


def real_wikipedia_html():
    errors = []
    for url in (ARTICLE, REST_HTML):
        try:
            request = Request(url, headers={
                "User-Agent": "NERAL-CPU-ReadPage-Audit/1.0 (research; no bot scraping)",
                "Accept": "text/html, application/xhtml+xml",
                "Accept-Language": "de",
                "Accept-Encoding": "identity",
            })
            with urlopen(request, timeout=18) as response:
                body = response.read(MAX_HTML_BYTES + 1)
                resolved = response.geturl()
                if (len(body) > MAX_HTML_BYTES or len(body) < 10000
                        or urlsplit(resolved).hostname != "de.wikipedia.org"):
                    raise ValueError("Wikipedia HTML missing or oversized")
                media = response.headers.get_content_type()
                if media not in ("text/html", "application/xhtml+xml"):
                    raise ValueError("Expected complete HTML, got another media type")
                text = body.decode(response.headers.get_content_charset() or "utf-8",
                                   errors="replace")
                if "Apollo 11" not in text and "Apollo_11" not in text:
                    raise ValueError("Unexpected Wikipedia article content")
                return text, resolved, url, len(body)
        except (OSError, ValueError) as exc:
            # The report never includes a URL query, private text or page body.
            errors.append(type(exc).__name__)
    raise RuntimeError("Real Wikipedia HTML unavailable: " + ",".join(errors))


def main():
    markup, resolved, selected, size = real_wikipedia_html()
    parser = MainText()
    parser.feed(markup)
    parser.close()
    # The MainText source parser consumes the complete real HTML; the
    # production 4,600-character excerpt is intentionally bounded.
    excerpt = parser.text(QUERY)
    full_visible = "".join((parser.infobox + parser.main)
                           if parser.main and parser.infobox
                           else (parser.main or parser.all))
    page = {
        "url": resolved, "page_read": True, "page_text": excerpt,
        "year_clip_code": parser.year_clip_code,
    }
    candidate = unique_read_year_source(QUESTION, [page])
    pipeline = year_source_pipeline_code(QUESTION, [page],
                                         {"web1": excerpt, "user": QUESTION})
    samples = []
    for match in list(YEAR.finditer(excerpt))[:12]:
        nearby = excerpt[max(0, match.start()-95): min(len(excerpt), match.end()+95)]
        samples.append({"year": match.group(0), "context": nearby.replace("\\n", " ")})
    report = {
        "cpu_only": True,
        "full_real_wikipedia_html_read": True,
        "source": "de.wikipedia.org",
        "source_variant": "canonical_html" if selected == ARTICLE else "wikipedia_rest_html",
        "html_bytes": size,
        "parsed_body_characters": len(full_visible),
        "extracted_characters": len(excerpt),
        "full_body_has_1961": bool(re.search(r"(?<!\d)1961(?!\d)", full_visible)),
        "full_body_has_1969": bool(re.search(r"(?<!\d)1969(?!\d)", full_visible)),
        "years_in_excerpt": sorted(set(YEAR.findall(excerpt))),
        "year_clip_code": parser.year_clip_code,
        "year_source_pipeline_code": pipeline,
        "candidate_year": candidate["year"] if candidate else None,
        "candidate_quote": candidate["quote"] if candidate else None,
        "year_context_samples": samples,
    }
    print("NERAL_APOLLO_REAL_HTML_CPU " + json.dumps(report, ensure_ascii=False))
    if candidate is None or candidate.get("year") != "1969" or pipeline != "candidate_available":
        raise AssertionError("Real Apollo 11 year evidence rejected or ambiguous")
    if candidate["quote"] not in excerpt:
        raise AssertionError("Quote does not literally occur in the read page")
    print("NERAL_APOLLO_REAL_HTML_CPU_PASS")


if __name__ == "__main__":
    main()
