"""First NERAL Dev Agent core: deterministic, bounded work continuation."""
import unittest
import json

from neral_dev_agent import (AgentError, start_task, next_state, execute_approved_task,
    task_snapshot, restore_task)


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



class FakeAgentTools:
    """Deterministic test double, NOT an AI model or GitHub access."""
    def __init__(self, results=(True,)):
        self.results=iter(results)
        self.called=[]
    def plan(self, task):
        self.called.append("plan")
        return "Plan: change only issue-scoped files; test first"
    def write_red_test(self, task):
        self.called.append("red")
        return "Expected failing unittest observed (mock fixture)"
    def apply_patch(self, task):
        self.called.append("patch")
        return "Fixture changed one file (mock fixture)"
    def run_tests(self, task):
        self.called.append("test")
        return next(self.results), "Simulated test result; not real CI"
    def create_draft_pr(self, task):
        self.called.append("pr")
        return "https://github.com/Cola968/neral-helios/pull/138"


class AutonomousContinuationTests(unittest.TestCase):
    def task(self):
        return start_task(repo="Cola968/neral-helios",issue=137,
            branch="agent/issue-137-automation",max_repairs=3)

    def test_one_approved_command_runs_plan_patch_tests_draft_pr_uninterrupted(self):
        tools=FakeAgentTools((False,False,True))
        state=execute_approved_task(self.task(),tools)
        self.assertEqual(state.phase,"WAITING_FOR_REVIEW")
        self.assertEqual(state.repair_attempts,2)
        self.assertEqual(tools.called,["plan","red","patch","test",
                                       "patch","test","patch","test","pr"])
        self.assertEqual(len(state.events),9)

    def test_repair_exhaustion_stops_before_pr_and_never_loops_forever(self):
        tools=FakeAgentTools((False,False,False,True))
        state=execute_approved_task(self.task(),tools)
        self.assertEqual(state.phase,"FAILED")
        self.assertEqual(state.repair_attempts,3)
        self.assertNotIn("pr",tools.called)
        self.assertEqual(tools.called.count("test"),3)

    def test_expired_deadline_stops_before_tool_access(self):
        calls=[]
        tools=FakeAgentTools()
        def expired_clock():
            calls.append(1)
            return 10 if len(calls)==1 else 20
        state=execute_approved_task(self.task(),tools,max_seconds=5,clock=expired_clock)
        self.assertEqual(state.phase,"FAILED")
        self.assertEqual(tools.called,[])
        self.assertEqual(state.events[-1][0],"deadline_exceeded")

    def test_adapter_crash_stops_without_leaking_exception_secret(self):
        tools=FakeAgentTools()
        def bad_plan(task):
            raise RuntimeError("hf_secret_12345 must not be in audit trail")
        tools.plan=bad_plan
        state=execute_approved_task(self.task(),tools)
        self.assertEqual(state.phase,"FAILED")
        self.assertNotIn("hf_secret",str(state))
        self.assertEqual(state.events[-1][0],"operation_failed")

    def test_missing_genuine_evidence_cannot_claim_a_step_passed(self):
        tools=FakeAgentTools()
        tools.run_tests=lambda task:(True,"")
        state=execute_approved_task(self.task(),tools)
        self.assertEqual(state.phase,"FAILED")
        self.assertNotIn("pr",tools.called)

    def test_wrong_repo_pr_is_rejected_and_not_presented_as_success(self):
        tools=FakeAgentTools()
        tools.create_draft_pr=lambda task:"https://github.com/attacker/repo/pull/123"
        state=execute_approved_task(self.task(),tools)
        self.assertEqual(state.phase,"FAILED")
        self.assertFalse(any(event=="draft_pr_opened" for event,_ in state.events))

    def test_unsafe_tool_result_type_and_invalid_budget_fails_closed(self):
        tools=FakeAgentTools()
        tools.run_tests=lambda task:("yes","tests passed")
        self.assertEqual(execute_approved_task(self.task(),tools).phase,"FAILED")
        for seconds in (0,3601,True,1.2,-4):
            with self.subTest(seconds=seconds),self.assertRaises(AgentError):
                execute_approved_task(self.task(),FakeAgentTools(),max_seconds=seconds)

    def test_partial_or_finished_tasks_cannot_restart_without_new_authorization(self):
        with self.assertRaises(AgentError):
            execute_approved_task(next_state(self.task(),"plan_ready",evidence="plan"),
                                  FakeAgentTools())
        with self.assertRaises(AgentError):
            execute_approved_task(next_state(self.task(),"cancel",evidence="cancelled"),
                                  FakeAgentTools())



class AgentResumeTests(unittest.TestCase):
    def task(self):
        return start_task(repo="Cola968/neral-helios",issue=137,
                          branch="agent/issue-137-resume",max_repairs=3)

    def test_round_trip_validated_history_can_be_read_after_service_restart(self):
        s=self.task()
        for event,proof in (
            ("plan_ready","plan"), ("red_test_observed","red failed"),
            ("patch_committed","patch SHA"),("tests_failed","red 2"),
            ("patch_committed","patch 2")):
            s=next_state(s,event,evidence=proof)
        saved=task_snapshot(s)
        self.assertLess(len(saved),16384)
        self.assertEqual(restore_task(saved),s)
        self.assertEqual(restore_task(saved).phase,"VERIFYING")
        self.assertEqual(restore_task(saved).repair_attempts,1)

    def test_corrupted_or_forged_state_is_not_accepted(self):
        s=next_state(self.task(),"plan_ready",evidence="approved plan")
        doc=json.loads(task_snapshot(s))
        for key,value in (
            ("phase","WAITING_FOR_REVIEW"),
            ("repair_attempts",2), ("branch","main"),
            ("repo","attacker/other"), ("issue",False)):
            bad=dict(doc)
            bad[key]=value
            with self.subTest(key=key), self.assertRaises(AgentError):
                restore_task(json.dumps(bad))
        with self.assertRaises(AgentError):
            restore_task(task_snapshot(s)[:-4])

    def test_unknown_fields_and_unbounded_data_rejected(self):
        saved=task_snapshot(self.task())
        malformed=json.loads(saved)
        malformed["some_unrecognized_permission"]="deploy_to_prod"
        with self.assertRaises(AgentError):
            restore_task(json.dumps(malformed))
        with self.assertRaises(AgentError):
            restore_task("x"*70000)

    def test_inconsistent_event_sequence_fails_closed(self):
        data=json.loads(task_snapshot(self.task()))
        data["events"]=[["tests_passed","the agent said it worked"]]
        data["phase"]="READY_FOR_DRAFT_PR"
        with self.assertRaises(AgentError):
            restore_task(json.dumps(data))


if __name__=="__main__":
    unittest.main()
