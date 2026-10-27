"""NERAL development agent v0: isolated control-plane state transitions.

This file is NOT a code-executing AI agent or a scheduler. It provides a
bounded and auditable state machine that a future permissioned cloud agent
can use to orchestrate multiple steps after a single approved assignment.
Only explicit events with evidence can move a task. No shell, HTTP, model
inference, GPU, credential access, or repository writes happen here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
import time
import json
import hashlib
import hmac
from typing import Protocol, Callable


class AgentError(ValueError):
    """Out-of-scope/invalid agent transition; caller must stop safely."""


_TERMINAL = frozenset({"CANCELLED", "FAILED", "WAITING_FOR_REVIEW"})
_ALLOWED = {
    ("PLANNING", "plan_ready"): "RED_TEST",
    ("RED_TEST", "red_test_observed"): "IMPLEMENTING",
    ("IMPLEMENTING", "patch_committed"): "VERIFYING",
    ("VERIFYING", "tests_passed"): "READY_FOR_DRAFT_PR",
    ("READY_FOR_DRAFT_PR", "draft_pr_opened"): "WAITING_FOR_REVIEW",
}
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_AGENT_BRANCH_RE = re.compile(r"agent/[A-Za-z0-9][A-Za-z0-9._/-]{0,95}\Z")


@dataclass(frozen=True)
class AgentTask:
    repo: str
    issue: int
    branch: str
    max_repairs: int
    phase: str = "PLANNING"
    repair_attempts: int = 0
    events: tuple[tuple[str, str], ...] = ()


def start_task(*, repo: str, issue: int, branch: str, max_repairs: int = 3) -> AgentTask:
    """Create scoped orchestration metadata; does not grant filesystem rights."""
    if not isinstance(repo, str) or not _REPO_RE.fullmatch(repo):
        raise AgentError("Repository must have owner/name format")
    if type(issue) is not int or issue <= 0:
        raise AgentError("Issue must be a positive GitHub issue number")
    if not isinstance(branch, str) or not _AGENT_BRANCH_RE.fullmatch(branch):
        raise AgentError("Agent may use only a restricted agent/* working branch")
    if "//" in branch or ".." in branch or branch.endswith("/") or branch.endswith("."):
        raise AgentError("Unsafe or ambiguous branch name")
    if type(max_repairs) is not int or not 1 <= max_repairs <= 3:
        raise AgentError("1–3 correction attempts are allowed")
    return AgentTask(repo=repo, issue=issue, branch=branch, max_repairs=max_repairs)


def _evidence(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise AgentError("Each transition requires short nonempty evidence")
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in value):
        raise AgentError("Control characters are not valid audit evidence")
    return value.strip()


def next_state(task: AgentTask, event: str, *, evidence: str) -> AgentTask:
    """Advance an approved task using evidence; no implicit extra work."""
    if not isinstance(task, AgentTask):
        raise AgentError("Expected AgentTask")
    if task.phase in _TERMINAL:
        raise AgentError("Terminal task cannot be resumed without new assignment")
    proof = _evidence(evidence)
    if event in ("operation_failed", "deadline_exceeded"):
        return replace(task, phase="FAILED", events=task.events + ((event,proof),))
    if event == "cancel":
        return replace(task, phase="CANCELLED", events=task.events + ((event,proof),))
    if event == "tests_failed" and task.phase == "VERIFYING":
        attempts = task.repair_attempts + 1
        phase = "FAILED" if attempts >= task.max_repairs else "IMPLEMENTING"
        return replace(task, phase=phase, repair_attempts=attempts,
                       events=task.events + ((event,proof),))
    destination = _ALLOWED.get((task.phase,event))
    if destination is None:
        raise AgentError(f"Invalid state transition: {task.phase} + {event}")
    if event == "draft_pr_opened":
        expected = "https://github.com/" + task.repo + "/pull/"
        suffix = proof[len(expected):] if proof.startswith(expected) else ""
        if not suffix.isdecimal() or int(suffix) < 1 or proof != expected + suffix:
            raise AgentError("Draft PR must be in the assigned repository")
    return replace(task, phase=destination, events=task.events + ((event,proof),))


class AgentTools(Protocol):
    """Only a separately authorized, sandboxed implementation may implement this.

    An AgentTools adapter is trusted integration code, NOT raw model output or
    untrusted issue instructions. Its evidence must correspond to real events.
    """
    def plan(self, task: AgentTask) -> str: ...
    def write_red_test(self, task: AgentTask) -> str: ...
    def apply_patch(self, task: AgentTask) -> str: ...
    def run_tests(self, task: AgentTask) -> tuple[bool, str]: ...
    def create_draft_pr(self, task: AgentTask) -> str: ...


class _Deadline(Exception):
    pass


def execute_approved_task(
    task: AgentTask,
    tools: AgentTools,
    *,
    max_seconds: int = 3600,
    clock: Callable[[], float] = time.monotonic,
) -> AgentTask:
    """Continue *one already approved task* through a bounded multi-step cycle.

    No implicit model call, arbitrary shell, GitHub API, persistent process or
    remote run. Real effects depend on an external trusted tool adapter, which
    MUST enforce per-step timeout, branch isolation, allowed paths, no secrets
    to generated code, budget cap, and review-before-merge independently.

    The deadline is checked between calls and DOES NOT interrupt a blocking
    adapter. Never connect this to unsandboxed arbitrary model commands.
    """
    if type(max_seconds) is not int or not 1 <= max_seconds <= 3600:
        raise AgentError("Workflow time limit must be between 1 and 3600 seconds")
    if not isinstance(task, AgentTask) or task.phase != "PLANNING" or task.events:
        raise AgentError("Only a fresh, explicitly approved scoped task can start")
    # Revalidate instead of trusting an attacker-controlled dataclass instance.
    if start_task(repo=task.repo, issue=task.issue,
                  branch=task.branch, max_repairs=task.max_repairs) != task:
        raise AgentError("Task metadata has been mutated; approval invalid")
    started = clock()

    def ensure_time():
        if clock() >= started + max_seconds:
            raise _Deadline

    def step(event: str, action):
        nonlocal task
        ensure_time()
        proof = action(task)
        ensure_time()
        task = next_state(task, event, evidence=proof)

    try:
        step("plan_ready", tools.plan)
        step("red_test_observed", tools.write_red_test)
        while task.phase == "IMPLEMENTING":
            step("patch_committed", tools.apply_patch)
            ensure_time()
            outcome = tools.run_tests(task)
            ensure_time()
            if not isinstance(outcome, tuple) or len(outcome) != 2 or type(outcome[0]) is not bool:
                raise AgentError("Tests must return (boolean_passed, evidence)")
            passed, proof = outcome
            task = next_state(task, "tests_passed" if passed else "tests_failed", evidence=proof)
        if task.phase == "READY_FOR_DRAFT_PR":
            step("draft_pr_opened", tools.create_draft_pr)
    except _Deadline:
        return next_state(task, "deadline_exceeded",
                          evidence="Workflow deadline exceeded; intervention required")
    except Exception:
        # Do not echo potentially sensitive adapter exceptions, issue text,
        # provider tokens, raw test output, or model prompts to audit evidence.
        return next_state(task, "operation_failed",
                          evidence="Adapter unavailable or invalid step evidence; intervention required")
    return task


_SNAPSHOT_FIELDS = frozenset({
    "version", "repo", "issue", "branch", "max_repairs",
    "phase", "repair_attempts", "events", "sha256",
})


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def task_snapshot(task: AgentTask) -> str:
    """Store task history on a *trusted* server; this does not grant authority.

    SHA-256 is an accidental-corruption checksum, NOT an authentication tag.
    Anyone who can modify snapshots and recompute their checksum can forge
    them. Never accept client snapshots as approval; use server-side storage.
    """
    if not isinstance(task, AgentTask) or len(task.events)>32:
        raise AgentError("Invalid or oversized agent task")
    _=start_task(repo=task.repo, issue=task.issue,
                 branch=task.branch, max_repairs=task.max_repairs)
    reconstructed=_
    for item in task.events:
        if not isinstance(item,tuple) or len(item)!=2:
            raise AgentError("Invalid task event history")
        reconstructed=next_state(reconstructed,item[0],evidence=item[1])
    if reconstructed!=task:
        raise AgentError("Task history does not match current phase and repair count")
    data={
        "version":1, "repo":task.repo, "issue":task.issue,
        "branch":task.branch, "max_repairs":task.max_repairs,
        "phase":task.phase, "repair_attempts":task.repair_attempts,
        "events":[list(item) for item in task.events],
    }
    checksum=hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()
    return _canonical_json({**data,"sha256":checksum})


def restore_task(snapshot: str) -> AgentTask:
    """Read a previously stored task; replay each transition before trusting phase."""
    if not isinstance(snapshot,str) or len(snapshot.encode("utf-8"))>16384:
        raise AgentError("Invalid agent task snapshot length")
    try:
        data=json.loads(snapshot)
        if not isinstance(data,dict) or set(data)!=_SNAPSHOT_FIELDS or data["version"]!=1:
            raise AgentError("Unsupported agent task snapshot")
        received=data["sha256"]
        if not isinstance(received,str) or not re.fullmatch(r"[0-9a-f]{64}",received):
            raise AgentError("Agent task checksum missing")
        unsigned={k:v for k,v in data.items() if k!="sha256"}
        expected=hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        if not hmac.compare_digest(received,expected):
            raise AgentError("Agent task snapshot changed after it was stored")
        task=start_task(repo=data["repo"],issue=data["issue"],branch=data["branch"],
                        max_repairs=data["max_repairs"])
        events=data["events"]
        if not isinstance(events,list) or len(events)>32:
            raise AgentError("Invalid agent event list")
        for entry in events:
            if (not isinstance(entry,list) or len(entry)!=2
                or not all(isinstance(v,str) for v in entry)):
                raise AgentError("Invalid agent event entry")
            task=next_state(task,entry[0],evidence=entry[1])
        if data["phase"]!=task.phase or data["repair_attempts"]!=task.repair_attempts:
            raise AgentError("Agent task sequence contradicts its claimed state")
    except (ValueError,TypeError,KeyError,OverflowError,RecursionError) as exc:
        raise AgentError("Agent snapshot validation failed") from exc
    return task
