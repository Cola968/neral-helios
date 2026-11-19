"""Gated V2.1 Windows localhost HTTP acceptance: noncanonical, CSV, web data.

No fake results. Every check calls the real HF/Docker service; an ambiguous
real-world scenario is checked for routing and real execution, not made-up gold.
"""
from __future__ import annotations
import json
import re
import secrets
from urllib.error import HTTPError, URLError
from urllib.request import Request

from scripts.neral_v2_live5 import BASE, Live


def safe_execution_metadata(result):
    """Only trusted status fields; never model text, file data or code."""
    if type(result) is not dict:
        return {"result_type":type(result).__name__}
    keys = ("tier", "model_generated", "verified", "safe_to_promote",
            "sandbox_executed", "tool_used", "tool_calls_succeeded",
            "sandbox_attempts", "verification_status",
            "auto_escalated_from", "model_origin", "adapter_loaded")
    safe = {}
    for key in keys:
        value = result.get(key)
        if type(value) in (bool, int) or (
                type(value) is str and len(value) <= 64):
            safe[key] = value
        elif value is None and key in result:
            safe[key] = None
    trace = result.get("sandbox_failure_trace")
    if type(trace) is list:
        safe["sandbox_failure_trace"] = [
            {key: value for key in ("stage", "error_type")
             if type(value := entry.get(key)) is str and len(value) <= 64}
            for entry in trace[:3] if type(entry) is dict
        ]
    return safe


def stdout_of(result):
    marker="Sandbox-Ausgabe (nicht unabhängig geprüft):\n"
    if marker not in str(result.get("text","")):
        raise ValueError("Exact Docker stdout absent")
    return result["text"].split(marker,1)[1].split("\n\n",1)[0].strip()


def verify(result, *, label, expected=None):
    if not (result.get("tier")=="Helios" and
            result.get("auto_escalated_from")=="Luna" and
            result.get("sandbox_executed") is True and
            result.get("tool_used")=="helios_code_sandbox" and
            result.get("verified") is False and
            result.get("safe_to_promote") is False and
            result.get("model_generated") is True and
            result.get("tool_calls_succeeded")==1 and
            1<=result.get("sandbox_attempts",0)<=3):
        raise ValueError("Real unverified Helios Docker execution absent for "+label)
    stdout=stdout_of(result)
    if expected is not None and not re.search(
            r"(?<!\d)"+re.escape(str(expected))+r"(?!\d)",stdout):
        raise ValueError("Wrong independently checked value for "+label+
                         "; synthetic stdout="+stdout[:90])
    return stdout


def multipart(api, question, raw:bytes):
    boundary="NERALV21"+secrets.token_hex(12)
    body=[]
    for key,value in (
            ("prompt",question),("history","[]"),("mode","generate"),
            ("tier","Luna"),("force_sandbox","true")):
        body.append(("--"+boundary+"\r\nContent-Disposition: form-data; name=\""+
                     key+"\"\r\n\r\n"+value+"\r\n").encode("utf-8"))
    body.append(("--"+boundary+"\r\nContent-Disposition: form-data; name=\"files\"; "+
                 "filename=\"daten.csv\"\r\nContent-Type: text/csv\r\n\r\n").encode("utf-8"))
    body.append(raw)
    body.append(("\r\n--"+boundary+"--\r\n").encode("ascii"))
    request=Request(BASE+"/api/neral/preview/attachment-chat",
                    data=b"".join(body),
                    headers={"Origin":BASE,"X-Neral-CSRF":api.csrf,
                             "Content-Type":"multipart/form-data; boundary="+boundary},
                    method="POST")
    with api.opener.open(request,timeout=380) as response:
        if response.status!=200:
            raise ValueError("Non-200 multipart HTTP")
        return json.loads(response.read(100000))


def main():
    api=Live()
    password="E2E-V21-"+secrets.token_hex(18)
    registered=False
    cleanup=False
    passed=[]
    failed=[]
    try:
        api.bootstrap()
        session=api.call("/api/neral/preview/session")
        if session.get("three_tier_enabled") is not True:
            raise ValueError("Three-tier service unavailable")
        api.csrf=session["csrf"]
        created=api.call("/api/neral/account/register",{
            "username":"e2ev21"+secrets.token_hex(7),"password":password})
        if created.get("authenticated") is not True:
            raise ValueError("Synthetic registration failed")
        registered=True
        api.csrf=created["csrf"]
        cases=[
            ("unusual_house",lambda:api.call("/api/neral/chat/three-tier",{
                "tier":"Luna","history":[],
                "prompt":("Lena, Tom und Mira teilen sich ein Ferienhaus für 840 €. "
                          "Lena reist nach der Hälfte der Zeit ab, Tom bleibt "
                          "die volle Woche und Mira bringt noch einen Gast für "
                          "die letzten zwei Tage mit; pro Person und Nacht "
                          "wird abgerechnet. Erkläre notwendige Annahmen.")}),None),
            ("csv_sum",lambda:multipart(api,
                "Summiere alle Werte der Spalte amount in der ausgewählten CSV.",
                b"person,amount\nA,2\nB,4\nC,6\n"),12),
            ("web_data_sum",lambda:api.call("/api/neral/chat/three-tier",{
                "tier":"Luna","history":[],"prompt":"Addiere die Beträge in den bereitgestellten Web-Suchergebnissen.",
                "web_data":[
                    {"title":"Monat A","snippet":"Betrag 2 Euro","url":"https://example.org/a"},
                    {"title":"Monat B","snippet":"Betrag 4 Euro","url":"https://example.org/b"}]}),6),
        ]
        for label,fn,expected in cases:
            result=None
            try:
                result=fn()
                value=verify(result,label=label,expected=expected)
                passed.append(label)
                print("NERAL_V21_LIVE_CASE",json.dumps({
                    "case":label,"pass":True,"sandbox_executed":True,
                    "verified":False,"stdout_preview":value[:100]},ensure_ascii=False))
            except Exception as exc:
                failed.append(label)
                print("NERAL_V21_LIVE_CASE",json.dumps({
                    "case":label,"pass":False,"error_type":type(exc).__name__,
                    "execution_metadata":safe_execution_metadata(result)},ensure_ascii=False))
    except (ValueError,KeyError,HTTPError,URLError) as exc:
        for label in ("unusual_house","csv_sum","web_data_sum"):
            if label not in passed and label not in failed:
                failed.append(label)
                print("NERAL_V21_LIVE_CASE",json.dumps({
                    "case":label,"pass":False,"error_type":type(exc).__name__}))
    finally:
        if registered:
            try:
                request=Request(BASE+"/api/neral/account",
                    data=json.dumps({"password":password}).encode("utf-8"),
                    headers={"Origin":BASE,"Content-Type":"application/json",
                             "X-Neral-CSRF":api.csrf},method="DELETE")
                with api.opener.open(request,timeout=15) as response:
                    cleanup=json.loads(response.read(10000)).get("deleted") is True
            except (HTTPError,URLError,ValueError,KeyError,OSError):
                pass
        else:
            cleanup=True
    success=(len(passed)==3 and not failed and cleanup)
    print("NERAL_V21_LIVE_RESULT",json.dumps({
        "pass":success,"cases_passed":len(passed),"cases_failed":len(failed),
        "account_deleted":cleanup,"model_answers_independently_verified":False,
    },sort_keys=True))
    return 0 if success else 1


if __name__=="__main__":
    raise SystemExit(main())
