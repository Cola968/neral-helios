"""Static regression checks: these legacy workflows must not start PC jobs on PRs."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"
WORKFLOWS = (
    "arctus350-clean-fp32-64step.yml",
    "arctus350-clean64-dev-eval.yml",
    "arctus350-exact-math-curriculum.yml",
    "arctus350-local-sanitize-splits.yml",
    "arctus350-reference-answer-audit.yml",
    "arctus350-training-readiness.yml",
)
PR_CPU_WORKFLOWS = {
    "arctus350-clean-fp32-64step.yml",
    "arctus350-exact-math-curriculum.yml",
    "arctus350-reference-answer-audit.yml",
    "arctus350-training-readiness.yml",
}
GUARD = "github.event_name == 'workflow_dispatch' && inputs.approve_pc == true"


def blocks(text):
    jobs = text.split("\njobs:\n", 1)[1]
    rows = re.split(r"(?m)^  ([a-z0-9-]+):\s*$", jobs)
    return dict(zip(rows[1::2], rows[2::2]))


class PcDispatchGuardTests(unittest.TestCase):
    def test_pc_requires_manually_selected_approval_that_defaults_false(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                events = text.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
                self.assertRegex(events, r"(?m)^  workflow_dispatch:$")
                self.assertRegex(events, r"(?m)^      approve_pc:$")
                self.assertRegex(events, r"(?m)^        type: boolean$")
                self.assertRegex(events, r"(?m)^        default: false$")
                self.assertRegex(events, r"(?m)^        required: true$")

    def test_every_self_hosted_job_requires_dispatch_and_explicit_true(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                jobs = blocks((ROOT / name).read_text(encoding="utf-8"))
                pc_jobs = [job for job in jobs.values() if re.search(
                    r"(?m)^    runs-on:.*self-hosted", job)]
                self.assertEqual(len(pc_jobs), 1)
                self.assertRegex(pc_jobs[0], r"(?m)^    if: " + re.escape(GUARD) + r"$")

    def test_existing_hosted_cpu_pr_checks_remain_available(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                events = text.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
                has_pr = bool(re.search(r"(?m)^  pull_request:$", events))
                self.assertEqual(has_pr, name in PR_CPU_WORKFLOWS)
                if has_pr:
                    self.assertTrue(any(
                        re.search(r"(?m)^    runs-on: ubuntu-latest$", job)
                        for job in blocks(text).values()
                    ))


if __name__ == "__main__":
    unittest.main()
