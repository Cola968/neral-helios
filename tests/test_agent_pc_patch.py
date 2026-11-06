"""Second PC scenario is a disposable coding benchmark, not a deploy."""
import json
import unittest
from scripts.neral_agent_pc_live_smoke import patch_smoke


class Fake:
    def __init__(self, choices):
        self.values = list(choices)
    def respond(self, messages):
        if not self.values:
            return json.dumps({"action":"finish","summary":"No more"})
        return json.dumps(self.values.pop(0))


class PatchSmokeTests(unittest.TestCase):
    def test_model_patch_is_checked_and_only_disposable_file_written(self):
        model = Fake([
            {"action":"read","path":"buggy.py"},
            {"action":"propose_edit","path":"buggy.py",
             "content":"def multiply(a, b):\n    return a * b\n"},
            {"action":"syntax_check"},
            {"action":"finish","summary":"Patched"},
        ])
        out = patch_smoke(model)
        self.assertTrue(out["patch_proposed"])
        self.assertTrue(out["patch_approved_test_only"])
        self.assertTrue(out["patch_ast_valid"])
        self.assertTrue(out["patch_logic_verified"])
        self.assertFalse(out["executed_generated_code_on_host"])

    def test_unrelated_file_patch_never_autoapproved(self):
        model = Fake([
            {"action":"propose_edit","path":"wrong.py",
             "content":"def multiply(a,b):return a*b\n"},
        ])
        out = patch_smoke(model)
        self.assertFalse(out["patch_approved_test_only"])
        self.assertFalse(out["patch_logic_verified"])
        self.assertFalse(out["executed_generated_code_on_host"])

    def test_no_fake_success_if_model_finishes_without_fix(self):
        out = patch_smoke(Fake([{"action":"finish","summary":"fixed"}]))
        self.assertFalse(out["patch_logic_verified"])
        self.assertFalse(out["patch_proposed"])
