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
    max_new_tokens: int = 256
    adapter_id: str | None = None


# Default Luna uses plain Qwen3.5:4B. Frozen PEFT Luna-v1 is only
# available for explicitly opted-in benchmark experiments.
# This advisory router does not itself load or activate any model.
LUNA_FAST_MAX_NEW_TOKENS = 96
LUNA_ADAPTER_ID = 'luna-v1-4b-peft'


LEVELS = ("Luna","Terra","Helios")
HEAVY = re.compile(r"\b(beweis|beweise|beweise? dass|z3|sympy|networkx|optimier|constraint|kombinatorik|graphfärb|sat|unsat|mehrstufig|docker|sandbox|komplexe logik)\b", re.I)
ONLINE = re.compile(r"\b(heute|aktuell|neueste|gerade|websuche|quellen|nachrichten|preis|wetter|202[6-9])\b",re.I)
FILES = re.compile(r"\b(pdf|datei|dokument|anhang|csv|repository|code|programmier|python|javascript|analyse)\b",re.I)
RISK = re.compile(r"\b(medizin|diagnos|juristisch|rechtsberatung|finanzberatung|selbstverletz|medikament)\b",re.I)


def choose_route(prompt: str, *, has_files: bool=False, force: str|None=None, uncertain: bool=False, previous_failure: bool=False, experimental_luna_adapter: bool=False) -> Route:
    if not isinstance(prompt,str) or len(prompt)>20000: raise ValueError("invalid prompt")
    if force not in (*LEVELS,None): raise ValueError("unknown requested tier")
    if type(experimental_luna_adapter) is not bool: raise ValueError("invalid adapter experiment flag")
    serious=bool(HEAVY.search(prompt)) or previous_failure
    context=bool(ONLINE.search(prompt) or FILES.search(prompt) or RISK.search(prompt) or has_files or uncertain)
    level=2 if serious else 1 if context or len(prompt)>240 else 0
    if force is not None: level=max(level,LEVELS.index(force))
    tier=LEVELS[level]
    tools=() if level==0 else ("web","rag","python") if level==1 else ("web","rag","python","z3","sympy","networkx","docker")
    return Route(tier, "qwen3.5:4b" if level==0 else "qwen3.5:9b", level==2,
                 tools, "complex" if serious else "context" if context else "simple",
                 96 if level==0 else 256 if level==1 else 768,
                 LUNA_ADAPTER_ID if level==0 and experimental_luna_adapter else None)


def escalate(route:Route, *, uncertain:bool=False, tool_failed:bool=False)->Route:
    if route.tier=="Helios": return route
    target = "Helios" if tool_failed else "Terra" if uncertain or route.tier=="Luna" else "Helios"
    if target=="Terra": return Route("Terra","qwen3.5:9b",False,("web","rag","python"),"escalation",256)
    return Route("Helios","qwen3.5:9b",True,("web","rag","python","z3","sympy","networkx","docker"),"escalation",768)


def generation_profile(route: Route) -> dict:
    """A strict generation profile for the selected tier, not a model switch.

    Actual model execution must separately confirm the chosen adapter bytes.
    Luna's PEFT fast path always disables thinking and limits generation.
    """
    if not isinstance(route, Route) or route.tier not in LEVELS:
        raise ValueError("Unknown route profile")
    expected = {"Luna": 96, "Terra": 256, "Helios": 768}[route.tier]
    if (route.max_new_tokens != expected or
        route.think is not (route.tier=="Helios") or
        (route.adapter_id is not None and
         not (route.tier == "Luna" and route.adapter_id == LUNA_ADAPTER_ID))):
        raise ValueError("Unsafe or inconsistent route profile")
    return {"model": route.base, "think": route.think,
            "options": {"num_predict": expected, "temperature": 0},
            "adapter_id": route.adapter_id}
