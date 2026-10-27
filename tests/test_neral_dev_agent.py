"""First NERAL Dev Agent core: deterministic, bounded work continuation."""
import unittest

from neral_dev_agent import AgentError, start_task, next_state


class AgentWorkflowTests(unittest.TestCase):
    def task(self, **overrides):
        kw={"repo":"Cola968/neral-helios","issue":137,
            "branch":"agent/issue-137-first-fix", "max_repairs":3}
        kw.update(overrides)
        return start_task(**kw)

    def test_full_issue_to_draft_pr_pipeline_without_new_user_input(self):
        state=self.task()
        self.assertEqual(state.phase,"PLANNING")
        self.assertEqual(state.repair_attempts,0)
        state=next_state(state,"plan_ready",evidence="Plan scoped to tests/foo.py")
        self.assertEqual(state.phase,"RED_TEST")
        state=next_state(state,"red_test_observed",evidence="unittest expected failure")
        self.assertEqual(state.phase,"IMPLEMENTING")
        state=next_state(state,"patch_committed",evidence="commit abcdef12")
        self.assertEqual(state.phase,"VERIFYING")
        state=next_state(state,"tests_passed",evidence="Run 123 green")
        self.assertEqual(state.phase,"READY_FOR_DRAFT_PR")
        state=next_state(state,"draft_pr_opened",
            evidence="https://github.com/Cola968/neral-helios/pull/138")
        self.assertEqual(state.phase,"WAITING_FOR_REVIEW")
        self.assertEqual(state.repair_attempts,0)
        self.assertEqual(len(state.events),5)

    def test_limit_repair_cycles_and_stop(self):
        state=next_state(self.task(),"plan_ready",evidence="plan")
        state=next_state(state,"red_test_observed",evidence="expected failure")
        for attempt in range(1,4):
            state=next_state(state,"patch_committed",evidence="patch")
            state=next_state(state,"tests_failed",evidence="error trace summary")
            self.assertEqual(state.repair_attempts,attempt)
            self.assertEqual(state.phase,"FAILED" if attempt==3 else "IMPLEMENTING")
        with self.assertRaises(AgentError):
            next_state(state,"patch_committed",evidence="should not run")

    def test_no_unauthorized_branch_writes(self):
        for branch in ("main","master","release","", "../main","refs/heads/main","agent/foo/../../main"):
            with self.subTest(branch=branch),self.assertRaises(AgentError):
                self.task(branch=branch)
        for issue in (0,-1,False,"137"):
            with self.subTest(issue=issue),self.assertRaises(AgentError):
                self.task(issue=issue)
        for repairs in (0,4,False,100):
            with self.subTest(repairs=repairs),self.assertRaises(AgentError):
                self.task(max_repairs=repairs)

    def test_require_real_evidence_and_correct_transition(self):
        with self.assertRaises(AgentError):
            next_state(self.task(),"tests_passed",evidence="all good")
        with self.assertRaises(AgentError):
            next_state(self.task(),"plan_ready",evidence="")
        s=next_state(self.task(),"plan_ready",evidence="plan")
        s=next_state(s,"red_test_observed",evidence="red proof")
        s=next_state(s,"patch_committed",evidence="patch")
        with self.assertRaises(AgentError):
            next_state(s,"tests_passed",evidence="")

    def test_pr_must_target_same_repository_and_cancellation_stops(self):
        s=next_state(self.task(),"plan_ready",evidence="plan")
        s=next_state(s,"red_test_observed",evidence="red")
        s=next_state(s,"patch_committed",evidence="patch")
        s=next_state(s,"tests_passed",evidence="green")
        with self.assertRaises(AgentError):
            next_state(s,"draft_pr_opened",evidence="https://github.com/other/repo/pull/123")
        with self.assertRaises(AgentError):
            next_state(s,"draft_pr_opened",evidence="https://github.com/Cola968/neral-helios/pull/138?token=abc")
        stopped=next_state(s,"cancel",evidence="human interrupted")
        self.assertEqual(stopped.phase,"CANCELLED")
        with self.assertRaises(AgentError):
            next_state(stopped,"draft_pr_opened",
                evidence="https://github.com/Cola968/neral-helios/pull/138")


if __name__=="__main__":
    unittest.main()
