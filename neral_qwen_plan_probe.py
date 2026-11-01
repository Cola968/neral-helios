"""Safe local Ollama availability check and one *proposal-only* model call.

This script NEVER downloads/pulls models, makes public web requests,
executes tools, edits files or persists model text. Only bounded metadata
is printed. A rejected Qwen plan is reported as rejected, not successful.
"""
from __future__ import annotations

import json
import urllib.request

from neral_agent_plan import propose_agent_plan

TAGS_URL="http://127.0.0.1:11434/api/tags"
MODEL="qwen3.5:9b"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


def installed_models():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    req=urllib.request.Request(TAGS_URL,headers={"Accept":"application/json"})
    try:
        with opener.open(req,timeout=3) as resp:
            if resp.status!=200:
                return []
            raw=resp.read(32769)
        if len(raw)>32768:
            return []
        payload=json.loads(raw)
        items=payload.get("models",[]) if isinstance(payload,dict) else []
        if not isinstance(items,list):
            return []
        return [row["name"] for row in items[:40]
                if isinstance(row,dict) and isinstance(row.get("name"),str)]
    except (OSError,ValueError,UnicodeError):
        return []


def inspect_probe(detect=None,planner=None):
    detect=installed_models if detect is None else detect
    planner=propose_agent_plan if planner is None else planner
    models=detect()
    available=isinstance(models,list) and MODEL in models
    base={"schema":1,"base_model":MODEL,"model_available":bool(available),
          "plan_validated":False,"execution_enabled":False,
          "tools_executed":False,"public_web_requested":False,
          "private_model_text_logged":False,"plan_persisted":False,
          "planning_only":True}
    if not available:
        return {**base,"status":"MODEL_UNAVAILABLE"}
    try:
        plan=planner(goal="Plane eine öffentliche Recherche zum Flughafen Berlin Brandenburg.",
                     model=MODEL)
        if (not isinstance(plan,dict) or plan.get("status")!="PROPOSED_UNAPPROVED"
            or plan.get("base_model")!=MODEL or
            plan.get("execution_enabled") is not False or
            not isinstance(plan.get("steps"),list) or
            plan.get("tools_executed")!=[]):
            raise ValueError("Invalid Qwen plan result")
        return {**base,"plan_validated":True,"status":"PLAN_VALIDATED",
                "step_count":len(plan["steps"])}
    except Exception:
        # No raw prompts, generated steps, tokens, local paths or exceptions
        # may leave the PC through GitHub Actions logs.
        return {**base,"status":"STRUCTURED_PLAN_REJECTED"}


def main():
    print(json.dumps(inspect_probe(),sort_keys=True,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
