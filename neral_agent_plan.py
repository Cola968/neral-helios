"""Explicitly requested Qwen3.5 agent PLAN ONLY, no executor or tools.

Uses the EXISTING neral_assistant.assistant_turn and its loopback-only
Ollama transport. Plans are untrusted candidate data; they cannot execute
shell commands, fetch URLs, mutate SQLite or trigger the Orbit worker.
A future reviewed integration may consume them after human approval.
"""
from __future__ import annotations

import json
import re

from neral_assistant import (
    assistant_turn, AssistantError, AssistantUnavailable, ALLOWED_MODELS,
)


class AgentPlanError(ValueError):
    """Reject malformed or unsafe unapproved model plans."""


_ALLOWED = frozenset({"research","review","draft"})
_BAD = re.compile(
    r"https?://|file://|\b(?:subprocess|powershell|cmd\.exe|bash -c|"
    r"os\.system|rm -rf|eval\(|exec\()",
    re.IGNORECASE,
)
_PROMPT = (
    "Du planst eine öffentliche Recherche, führst aber KEINE Aktionen aus. "
    "Erzeuge AUSSCHLIESSLICH ein JSON-Objekt ohne weitere Aussagen mit genau "
    "dem Schlüssel 'steps' und 3 bis 4 Elementen. Jedes Element enthält "
    "nur 'type' (research, review oder draft) und 'instruction' (12–160 Zeichen). "
    "Der erste Schritt ist research, mindestens einer ist review, der letzte "
    "draft. Schlage keine Shell-Befehle, Dateien, URL-Aufrufe, Änderungen, "
    "Anmeldungen oder Hintergrundjobs vor. Eine spätere Ausführung erfordert "
    "separates menschliches Einverständnis. "
    "Auftrag: "
)


def _parse(raw):
    if not isinstance(raw,str) or len(raw)>5000:
        raise AgentPlanError("Lokaler Agentenplan ist nicht gültig.")
    text=raw.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text=text[len("```json\n"):-len("\n```")].strip()
    try:
        plan=json.loads(text,parse_constant=lambda _: (_ for _ in ()).throw(ValueError("No non-finite JSON")))
    except (ValueError,TypeError) as exc:
        raise AgentPlanError("Qwen lieferte keinen gültigen JSON-Aufgabenplan.") from exc
    if not isinstance(plan,dict) or set(plan)!={"steps"}:
        raise AgentPlanError("Nicht erlaubte Modell-Planfelder.")
    steps=plan["steps"]
    if not isinstance(steps,list) or not 3<=len(steps)<=4:
        raise AgentPlanError("Ein Agentenplan muss 3 bis 4 Schritte enthalten.")
    if not all(isinstance(step,dict) for step in steps):
        raise AgentPlanError("Ungültige Modellschritte.")
    if steps[0].get("type")!="research" or steps[-1].get("type")!="draft":
        raise AgentPlanError("Agentenplan hat unzulässige Abfolge.")
    if not any(row.get("type")=="review" for row in steps):
        raise AgentPlanError("Quellenprüfung fehlt.")
    safe=[]
    for step in steps:
        if not isinstance(step,dict) or set(step)!={"type","instruction"}:
            raise AgentPlanError("Nicht freigegebene Aktionsfelder.")
        typ,action=step["type"],step["instruction"]
        if (typ not in _ALLOWED or not isinstance(action,str) or
                not 12<=len(action.strip())<=160 or
                any(ord(ch)<32 or ord(ch)==127 for ch in action) or
                _BAD.search(action)):
            raise AgentPlanError("Modell schlägt eine nicht erlaubte Aktion vor.")
        safe.append({"type":typ,"instruction":action.strip()})
    return safe


def propose_agent_plan(goal, *, model="qwen3.5:9b", transport=None):
    if (not isinstance(goal,str) or not 12<=len(goal.strip())<=400 or
        any(ord(ch)<32 or ord(ch)==127 for ch in goal)):
        raise AgentPlanError("Auftrag muss 12 bis 400 Zeichen ohne Steuerzeichen haben.")
    if model not in ALLOWED_MODELS:
        raise AgentPlanError("Unzulässige Qwen-Basis.")
    try:
        result=assistant_turn(
            _PROMPT+goal.strip(),transport=transport,model=model,think=False,
            enable_tools=False,temperature=0.0,max_new=320,
        )
    except AssistantUnavailable:
        raise
    except AssistantError as exc:
        raise AgentPlanError("Lokaler Qwen-Planungsaufruf abgelehnt.") from exc
    if (result.get("base_model")!=model or result.get("source")!="external-model"
        or result.get("tools_used")!=[] or result.get("model_generated") is not True):
        raise AgentPlanError("Unpassende Modellherkunft oder unerlaubte Ausführung.")
    steps=_parse(result.get("text"))
    return {
        "schema":1,"status":"PROPOSED_UNAPPROVED",
        "steps":steps,
        "source":"external-model","base_model":model,
        "model_origin":"Qwen3.5 via local Ollama, not native Arctus350",
        "requires_human_approval":True,
        "execution_enabled":False,
        "tools_executed":[],
        "web_requests_made":False,
        "model_generated":True,
        "orbit_integration_ready":False,
    }
