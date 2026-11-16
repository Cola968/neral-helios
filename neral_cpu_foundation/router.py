"""Advisory three-tier routing; never a proof or authorization decision."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Route:
    tier: str
    base: str
    think: bool
    tools: tuple[str,...]
    reason: str


LEVELS = ("Luna","Terra","Helios")
HEAVY = re.compile(r"\b(beweis|beweise|beweise? dass|z3|sympy|networkx|optimier|constraint|kombinatorik|graphfärb|sat|unsat|mehrstufig|docker|sandbox|komplexe logik)\b", re.I)
ONLINE = re.compile(r"\b(heute|aktuell|neueste|gerade|websuche|quellen|nachrichten|preis|wetter|202[6-9])\b",re.I)
FILES = re.compile(r"\b(pdf|datei|dokument|anhang|csv|repository|code|programmier|python|javascript|analyse)\b",re.I)
RISK = re.compile(r"\b(medizin|diagnos|juristisch|rechtsberatung|finanzberatung|selbstverletz|medikament)\b",re.I)


def choose_route(prompt: str, *, has_files: bool=False, force: str|None=None, uncertain: bool=False, previous_failure: bool=False) -> Route:
    if not isinstance(prompt,str) or len(prompt)>20000: raise ValueError("invalid prompt")
    if force not in (*LEVELS,None): raise ValueError("unknown requested tier")
    serious=bool(HEAVY.search(prompt)) or previous_failure
    context=bool(ONLINE.search(prompt) or FILES.search(prompt) or RISK.search(prompt) or has_files or uncertain)
    level=2 if serious else 1 if context or len(prompt)>240 else 0
    if force is not None: level=max(level,LEVELS.index(force))
    tier=LEVELS[level]
    tools=() if level==0 else ("web","rag","python") if level==1 else ("web","rag","python","z3","sympy","networkx","docker")
    return Route(tier, "qwen3.5:4b" if level==0 else "qwen3.5:9b", level==2, tools, "complex" if serious else "context" if context else "simple")


def escalate(route:Route, *, uncertain:bool=False, tool_failed:bool=False)->Route:
    if route.tier=="Helios": return route
    target = "Helios" if tool_failed else "Terra" if uncertain or route.tier=="Luna" else "Helios"
    if target=="Terra": return Route("Terra","qwen3.5:9b",False,("web","rag","python"),"escalation")
    return Route("Helios","qwen3.5:9b",True,("web","rag","python","z3","sympy","networkx","docker"),"escalation")
