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
