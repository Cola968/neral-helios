"""Bounded, double-confirmed NERAL Orbit + local Qwen one-shot composition.

Reuse the EXISTING modules from Draft #176 (OrbitStore and Wikipedia) and
Draft #203 (Qwen Assistant); do not silently copy/mimic either engine.
No arbitrary agent tools, loops, scheduling, publishing or uploads.
"""
from __future__ import annotations


class AgentLoopError(ValueError):
    """This task was rejected or failed; never fabricate a successful report."""


def run_confirmed_research(goal, *, confirm_model,confirm_public,plan,research,draft,store):
    if confirm_model is not True or confirm_public is not True:
        raise AgentLoopError("Human approval for local model and public research is required")
    if not isinstance(goal,str) or not 12<=len(goal.strip())<=200:
        raise AgentLoopError("Invalid public research goal")
    proposal=plan(goal.strip())
    if (not isinstance(proposal,dict) or proposal.get("status")!="PROPOSED_UNAPPROVED"
        or proposal.get("execution_enabled") is not False or
        proposal.get("base_model")!="qwen3.5:9b" or
        proposal.get("tools_executed")!=[] or
        not isinstance(proposal.get("steps"),list) or
        not 3<=len(proposal["steps"])<=4):
        raise AgentLoopError("Qwen proposal did not pass human-gated plan contract")
    kinds=[s.get("type") for s in proposal["steps"] if isinstance(s,dict)]
    if (len(kinds)!=len(proposal["steps"]) or kinds[0]!="research"
        or "review" not in kinds or kinds[-1]!="draft"
        or not set(kinds).issubset({"research","review","draft"})):
        raise AgentLoopError("Unapproved or invalid model-directed tool sequence")
    # Only this one fixed, previously approved public provider is callable.
    task=store.create(kind="research",title="Orbit: öffentliche Quellenrecherche",
                      goal=goal.strip())
    approved=store.transition(task["id"],"approve",revision=task["revision"])
    running=store.begin_research(task["id"],revision=approved["revision"])
    try:
        report=research(goal.strip())
        if (not isinstance(report,dict) or
            report.get("provider")!="Wikipedia (de)" or
            report.get("requires_review") is not True or
            report.get("model_used") is not False or
            not isinstance(report.get("sources"),list) or
            not 1<=len(report["sources"])<=3 or
            report.get("source_count")!=len(report["sources"])):
            raise AgentLoopError("Untrusted source contract rejected")
    except Exception as exc:
        store.finish_research(task["id"],revision=running["revision"],
                              error="Öffentliche Recherche fehlgeschlagen.")
        raise AgentLoopError("Public research failed; no success fabricated") from exc
    stored=store.finish_research(task["id"],revision=running["revision"],report=report)
    # Drafting can fail without destroying the already verified source report.
    # The final text is NEVER auto-published, logged or transferred remotely.
    try:
        response=draft(report["sources"])
        drafted=isinstance(response,str) and bool(response.strip()) and len(response)<=8000
    except Exception:
        drafted=False
    return {
        "schema":1,"status":stored["status"],
        "plan_validated":True,"plan_step_count":len(kinds),
        "research_completed":True,
        "source_count":report["source_count"],
        "draft_generated":bool(drafted),
        "requires_human_review":True,
        "auto_published":False,
        "agent_iterations":1,"public_research_requests":1,
        "tools_executed":["wikipedia_public_search_once"],
        "model_origin":"Qwen3.5 via local Ollama; Arctus350 not used",
        "model_weights_modified":False,
        "private_data_uploaded":False,
    }


def draft_with_qwen(sources):
    """One local Qwen-only text draft from bounded public excerpts, not a source of truth."""
    from neral_assistant import assistant_turn
    if not isinstance(sources,list) or not 1<=len(sources)<=3:
        raise AgentLoopError("Invalid public source list")
    lines=[]
    for i,row in enumerate(sources,1):
        if (not isinstance(row,dict) or
            not isinstance(row.get("excerpt"),str) or len(row["excerpt"])>700 or
            not isinstance(row.get("title"),str) or len(row["title"])>180):
            raise AgentLoopError("Invalid source text")
        lines.append(f"Quelle {i}: {row['title']}\n{row['excerpt']}")
    prompt=("Fasse die folgenden öffentlichen Wikipedia-Auszüge vorsichtig auf Deutsch "
            "zusammen. Ungeprüfte Angaben ausdrücklich kennzeichnen. "
            "Keine zusätzlichen Fakten oder erfundene Quellen. "
            "Dies ist nur ein unpublizierter Entwurf zur menschlichen Prüfung.\n"
            +"\n".join(lines))
    result=assistant_turn(prompt,model="qwen3.5:9b",enable_tools=False,
                          think=False,temperature=0.0,max_new=300)
    return result["text"]
