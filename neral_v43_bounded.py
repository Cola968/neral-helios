"""Opt-in V4.3 draft-first controller. CPU coordination, no GPU imports.

Callbacks are trusted runtime integrations, never model-supplied commands.
The caller owns the cooperative request deadline and resource isolation.
"""
from dataclasses import dataclass
from neral_v42_deadline import DeadlineExceeded, remaining_seconds


@dataclass(frozen=True)
class Verdict:
    status: str
    evidence: str = ""
    sources: tuple = ()


def enabled():
    import os
    return os.environ.get("NERAL_V43_BOUNDED", "0") == "1"


def bounded_answer(generate, verify=None, *, repair=None, remaining=remaining_seconds):
    rounds=0
    def has_time(minimum):
        left=remaining()
        return left is None or left>=minimum
    def valid(answer):
        return (isinstance(answer,dict) and isinstance(answer.get("text"),str)
                and bool(answer["text"].strip()) and answer.get("error_code") is None)
    if not has_time(1):
        raise DeadlineExceeded("neral_request_deadline")
    rounds+=1
    draft=generate()
    if not valid(draft):
        raise RuntimeError("No complete initial answer")
    draft=dict(draft)
    found_sources=()
    def result(candidate,status,verified=False):
        incomplete=status in ("verification_incomplete","counterexample_unresolved")
        warning=("Ein Gegenbeispiel wurde gefunden; die Korrektur konnte nicht bestätigt werden."
                 if status=="counterexample_unresolved" else
                 "Zusatzprüfung nicht abgeschlossen; Antwort ist ungeprüft." if incomplete else None)
        return {**candidate,"error_code":None,"verified":verified,
                "safe_to_promote":False,"agent_checked":verified,
                "tool_verified":verified,"proof_backend":None,
                "verification_scope":"bounded_independent_check" if verified else "none",
                "verification_status":status,"verification_warning":warning,
                "retrieved_sources":list(found_sources),
                "generation_rounds":rounds,"repair_attempts":rounds-1,
                "repair_success":rounds==2 and verified,"v43_bounded":True}
    if verify is None:
        return result(draft,"verification_not_requested")
    def evaluate(candidate):
        if not has_time(2):
            return Verdict("timeout")
        try:
            verdict=verify(dict(candidate))
        except (TimeoutError,RuntimeError,OSError,ValueError):
            return Verdict("timeout" if not has_time(2) else "unknown")
        if not isinstance(verdict,Verdict):
            return Verdict("unknown")
        return verdict
    verdict=evaluate(draft)
    found_sources=verdict.sources
    if verdict.status=="verified":
        return result(draft,"verified",True)
    if verdict.status!="counterexample":
        return result(draft,"verification_not_applicable" if verdict.status=="unsupported"
                      else "verification_incomplete")
    if repair is None or not verdict.evidence or not has_time(12):
        return result(draft,"counterexample_unresolved")
    rounds+=1
    try:
        candidate=repair(dict(draft),verdict.evidence)
    except (TimeoutError,RuntimeError,OSError,ValueError):
        return result(draft,"counterexample_unresolved")
    if valid(candidate) and evaluate(candidate).status=="verified":
        return result(candidate,"verified",True)
    return result(draft,"counterexample_unresolved")


def exact_integer_verdict(candidate, certified):
    """Compare only unambiguous full integers or explicit final-value markers.

    A certificate comes from trusted independent source processing. Never
    interpret an arbitrary number appearing in the middle of an explanation.
    """
    import re
    from neral_v42_quality import final_marked_value
    def extract(text):
        if not isinstance(text,str): return None
        value=final_marked_value(text) if "ENDERGEBNIS" in text.upper() else text
        if not isinstance(value,str): return None
        value=value.strip().strip("`").strip()
        return int(value) if re.fullmatch(r"[-+]?\d{1,30}",value) else None
    a,b=extract(candidate.get("text")),extract(certified.get("text"))
    if a is None or b is None: return Verdict("unsupported")
    if a==b: return Verdict("verified")
    return Verdict("counterexample","Independent exact integer result: "+str(b))
