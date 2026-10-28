"""Day-2 contract (2026-10-29) is explicit and the agent refuses invalid refs.

These are contract and CPU-only safety tests, not end-to-end agent proof.
"""
from pathlib import Path
import unittest

from neral_dev_agent import AgentError, start_task, next_state

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"NERAL_DAY2_PRODUCT_CONTRACT.md"


class Day2ProductContractTests(unittest.TestCase):
    def test_single_real_user_request_has_criteria_for_chat_and_coding_agent(self):
        text=CONTRACT.read_text(encoding="utf-8")
        for requirement in (
            "Ohne jede weitere Nachricht", "AC-01", "AC-02", "AC-03", "AC-04",
            "AC-05", "AC-06", "AC-07", "AC-08", "AC-09", "AC-10",
            "NERAL-native", "Qwen", "Arctus350",
            "maximal 3", "60 Minuten", "WAITING_FOR_REVIEW",
            "kein automatisch", "0 €", "GitHub-hosted",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement,text)
        self.assertIn("NICHT implementiert",text)
        self.assertIn("messbarer Abnahmetest",text)

    def test_scope_branch_policy_accepts_only_valid_nested_git_refs(self):
        for branch in (
            "agent/fix-input-validation",
            "agent/issue-137-work",
            "agent/case/step_2",
        ):
            with self.subTest(branch=branch):
                self.assertEqual(start_task(
                    repo="Cola968/neral-helios",issue=137,
                    branch=branch).branch,branch)

    def test_git_invalid_dot_components_and_lock_refs_fail_closed(self):
        for branch in (
            "agent/foo/.git",
            "agent/foo/.hidden/bar",
            "agent/foo/./bar",
            "agent/foo/bar.lock",
            "agent/a.lock/issue-137",
            "agent/a..b", "agent/a//b",
        ):
            with self.subTest(branch=branch):
                with self.assertRaises(AgentError):
                    start_task(repo="Cola968/neral-helios",issue=137,branch=branch)

    def test_contract_does_not_claim_a_real_coding_agent_when_only_state_machine_runs(self):
        task=start_task(repo="Cola968/neral-helios",issue=137,branch="agent/issue-137-demo")
        planned=next_state(task,"plan_ready",evidence="Spec prepared, no external side effects")
        self.assertEqual(planned.phase,"RED_TEST")
        self.assertEqual(planned.repair_attempts,0)
        self.assertEqual(planned.events[0][0],"plan_ready")
        with self.assertRaises(AgentError):
            next_state(planned,"draft_pr_opened",evidence="https://github.com/Cola968/neral-helios/pull/138")


if __name__=="__main__":unittest.main()
