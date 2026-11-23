"""Only public CPU web/Kiwix read diagnostics. Never imports model backends.

Isolated branch only, pinned NERAL production code under ./pinned.
No private datasets, no GPU usage, no service owner changes, no page-text logs.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import socket
import sys
from urllib.parse import urlsplit

ROOT = Path(os.environ["NERAL_PINNED_SOURCE"]).resolve()
sys.path.insert(0, str(ROOT))

from neral_local_wiki import local_wiki_search
from neral_web_search import WebSearch, WebSearchError
from neral_page_reader import read_top_pages
from neral_v42_quality import (
    _YEAR_TOKEN, _YEAR_EVENT, _year_event_subject,
    _read_page_subject_heading, _year_quote_supports,
    _subject_mismatch_near_year_window,
    unique_read_year_source, year_source_pipeline_code,
)


def emit(tag, data):
    print(tag + " " + json.dumps(data, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":")), flush=True)


def real_diag_query():
    src = (ROOT / "scripts" / "neral_v42_telemetry_probe.py").read_text("utf-8")
    module = ast.parse(src)
    expr = next(
        stmt.value for stmt in module.body
        if isinstance(stmt, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == "DIAGNOSTIC_CASES"
            for t in stmt.targets)
    )
    prompts = ast.literal_eval(expr)
    assert len(prompts) == 3 and prompts[2][0] == "knowledge"
    question = prompts[2][1]
    query = "Apollo 11 erste bemannte Mondlandung Jahr NASA"
    assert query in src and "Apollo 11" in question
    return question, query


def split_exactly_as_verifier(text):
    cut = re.compile(r"(?<=[.!?])(?:[ \t]+|\r?\n+)|\r?\n+")
    months = re.compile(
        r"(?iu)^[ \t]+(?:Januar|Februar|März|Maerz|April|Mai|"
        r"Juni|Juli|August|September|Oktober|November|Dezember)\b"
    )
    spans = []
    left = 0
    for match in cut.finditer(text):
        if (match.start() >= 2 and
                text[match.start()-1] == "." and
                text[match.start()-2].isdigit() and
                months.match(text[match.start():])):
            continue
        seg = text[left:match.start()]
        if seg.strip():
            spans.append((left + len(seg)-len(seg.lstrip()),
                          left + len(seg.rstrip())))
        left = match.end()
    if text[left:].strip():
        seg = text[left:]
        spans.append((left + len(seg)-len(seg.lstrip()),
                      left + len(seg.rstrip())))
    return spans


def evaluate_windows(question, source):
    text = source["page_text"]
    event, subject = _year_event_subject(question)
    spans = split_exactly_as_verifier(text)
    heading = _read_page_subject_heading(subject, text)
    windows = []
    all_years = sorted(set(_YEAR_TOKEN.findall(text)))
    for i, (start, _) in enumerate(spans):
        for width in (1, 2, 3):
            if i+width > len(spans):
                continue
            last = text[spans[i+width-1][0]:spans[i+width-1][1]]
            segment = text[start:spans[i+width-1][1]].strip()
            years = sorted(set(_YEAR_TOKEN.findall(segment)))
            if not years:
                continue
            verbs = sorted(set(_YEAR_EVENT[event].findall(segment)))
            mission_names = sorted(set(
                m.casefold() for m in
                re.findall(r"(?i)\bapollo\s+\d{1,3}\b", segment)))
            negation = bool(re.search(
                r"(?iu)\b(?:nicht|never|did\s+not|not\s+land|"
                r"geplant|planned|sollte)\b", segment))
            last_ok = (width < 3 or
                       (_YEAR_TOKEN.search(last) is not None and
                        _YEAR_EVENT[event].search(last) is not None))
            size_ok = 25 <= len(segment) <= 400
            support = (len(years) == 1 and size_ok and last_ok and
                       _year_quote_supports(
                           question, segment, years[0], page_text=text))
            nearby = (support and _subject_mismatch_near_year_window(
                question, segment, text, spans, i))
            allowed = bool(support and not nearby)
            blocker = ("accepted" if allowed else
                       "window_outside_25_400" if not size_ok else
                       "3_sentence_last_missing_year_event" if not last_ok else
                       "multiple_years" if len(years) != 1 else
                       "event_verb_missing" if not verbs else
                       "other_mission_in_window" if any(
                           m != subject.casefold() for m in mission_names) else
                       "nearby_competing_mission" if nearby else
                       "subject_or_semantics_failed")
            score = (100 if allowed else 0) + (20 if verbs else 0) + (
                15 if subject.casefold() in segment.casefold() else 0) + (
                5 if heading else 0) - width
            windows.append({
                "span_start":start,
                "span_end":spans[i+width-1][1],
                "window_span":{"sentences":width,"char_length":len(segment),
                               "segment_index":i},
                "has_subject":subject.casefold() in segment.casefold(),
                "read_page_heading_subject":heading,
                "matched_event_verbs":verbs[:6],
                "years_found":years[:6],
                "missions_found":mission_names[:6],
                "has_negation":negation,
                "three_sentence_last_ok":last_ok,
                "size_25_400_ok":size_ok,
                "exact_verifier_support":bool(support),
                "nearby_subject_veto":bool(nearby),
                "accepted":allowed,
                "failure_condition":blocker,
                "_score":score,
            })
    wins=sorted(windows,key=lambda x:(-x["_score"], x["span_start"],
                                      x["window_span"]["sentences"]))
    for win in wins:
        win.pop("_score",None)
    return {
        "page_segment_count":len(spans),
        "page_years_found":all_years[:14],
        "read_page_heading_subject":heading,
        "year_containing_windows":len(windows),
        "accepted_windows":sum(w["accepted"] for w in windows),
        "top_year_windows":wins[:6],
    }


def kiwix_read(question):
    try:
        with socket.create_connection(("127.0.0.1", 9691),timeout=1):
            running=True
    except OSError:
        running=False
    emit("NERAL_APOLLO_KIWIX_STATUS",{"port_9691_reachable":running,
                                      "used_in_3x3_web_query_route":False})
    if not running:
        return
    for label,text in [
        ("exact_public_diagnostic_prompt",question),
        ("apollo_keyword_control","Apollo 11 Mondlandung"),
    ]:
        try:
            rows=local_wiki_search(text)
            clean=[{
                "title":str(x.get("title",""))[:110],
                "book":str(x.get("book",""))[:85],
                "path":urlsplit(str(x.get("url",""))).path[:200],
                "context_has_1969":bool(re.search(r"\b1969\b",str(x.get("context","")))),
                "context_has_landing_word":bool(_YEAR_EVENT["land"].search(
                    str(x.get("context","")))),
            } for x in rows]
            emit("NERAL_APOLLO_KIWIX_READ",{"query_kind":label,"hits":clean,
                                           "count":len(clean)})
        except Exception as exc:
            emit("NERAL_APOLLO_KIWIX_READ",{"query_kind":label,
                                           "error_type":type(exc).__name__})


def web_read(question, query):
    # Exactly the authorized public Web V4 search query, same read_top_pages.
    search=WebSearch()
    try:
        result=search.search(query)
    except (WebSearchError,OSError,ValueError,TypeError) as exc:
        emit("NERAL_APOLLO_WEB_SEARCH",{"error_type":type(exc).__name__,
                                        "results":0})
        return
    hits=result.get("results",[])
    emit("NERAL_APOLLO_WEB_SEARCH",{
        "provider":str(result.get("provider","unknown"))[:90],
        "search_hit_count":len(hits),
        "top_search_hits":[{"rank":i+1,
                            "title":str(x.get("title",""))[:150],
                            "path":urlsplit(str(x.get("url",""))).path[:200],
                            "host":urlsplit(str(x.get("url",""))).hostname}
                           for i,x in enumerate(hits[:5])],
    })
    sources=read_top_pages(hits,query)
    readable=sum(s.get("page_read") is True for s in sources)
    limit=min(3000,6000//max(1,readable))
    literal=unique_read_year_source(question,sources)
    evidence={}
    for i,s in enumerate(sources,1):
        if s.get("page_read") is not True:
            continue
        content=s.get("page_text","")
        excerpt=content[:limit]
        if (literal and literal["source"] == f"web{i}" and
                literal["quote"] not in excerpt and
                len(literal["quote"]) < limit):
            quote=literal["quote"]
            excerpt=quote+"\n"+excerpt[:limit-len(quote)-1]
        evidence[f"web{i}"]=excerpt
    emit("NERAL_APOLLO_GLOBAL_CHECK",{
        "pages_considered":len(sources),
        "pages_read":readable,
        "literal_year_candidate_found":literal is not None,
        "evidence_budget_per_page":limit,
        "year_source_pipeline_code":year_source_pipeline_code(
            question,sources,evidence),
    })
    for i,src in enumerate(sources,1):
        url=str(src.get("url",""))
        data={
            "rank":i,
            "search_result_title":str(src.get("title",""))[:150],
            "opened_host":urlsplit(url).hostname,
            "opened_path":urlsplit(url).path[:220],
            "page_read":src.get("page_read") is True,
            "year_clip_code":src.get("year_clip_code"),
        }
        if src.get("page_read") is True:
            content=src.get("page_text","")
            data.update({
                "extracted_chars":len(content),
                "page_candidate_found":unique_read_year_source(
                    question,[src]) is not None,
                "candidate_quote_in_evidence":(
                    literal is not None and literal["source"] == f"web{i}"
                    and literal["quote"] in evidence.get(f"web{i}","")),
            })
            data.update(evaluate_windows(question,src))
        emit("NERAL_APOLLO_READ_PAGE",data)


def main():
    question,query=real_diag_query()
    emit("NERAL_APOLLO_CPU_DIAG_BEGIN",{
        "pinned_source_sha":"48c641aea7a3e9e2f9bc0ffc4d9a85dd6545b76e",
        "gpu_inference":False,
        "no_private_or_sealed_data":True,
        "prompt_type":"exact_approved_public_apollo_year",
        "model_imports":False,
        "query_type":"web_v4_approved_search",
    })
    kiwix_read(question)
    web_read(question,query)
    emit("NERAL_APOLLO_CPU_DIAG_COMPLETE",{
        "no_model_inference":True,
        "no_benchmark":True,
        "no_private_data":True,
    })


if __name__=="__main__":
    main()
