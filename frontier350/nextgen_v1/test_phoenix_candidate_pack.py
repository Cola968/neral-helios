import tempfile
import unittest
from pathlib import Path

import phoenix_candidate_pack as pack


class CandidateAuditTests(unittest.TestCase):
    def test_audit_counts_duplicate_answer_as_hint_only(self):
        splits = {"train": [
            {"id":"a","source_id":"x","user":"A sufficiently unique question one","assistant":"This is a sufficiently long answer that is useful for testing and contains enough words."},
            {"id":"b","source_id":"x","user":"A sufficiently unique question two","assistant":"This is a sufficiently long answer that is useful for testing and contains enough words."},
        ], "validation": [
            {"id":"c","source_id":"x","user":"A sufficiently unique question three","assistant":"Another sufficiently long response that avoids the structural hard filters in the audit."}
        ]}
        result = pack.audit(splits)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["flags"].get("duplicate_answer"), 1)
        self.assertEqual(result["distinct_prompts"], 3)

    def test_audit_detects_hard_privacy_role_and_length_flags(self):
        splits = {"train": [{"id":"a","source_id":"x","user":"mail a@example.com","assistant":"Assistant: short"}], "validation": []}
        result = pack.audit(splits)
        self.assertEqual(result["flags"].get("email_pattern"), 1)
        self.assertEqual(result["flags"].get("role_marker"), 1)
        self.assertEqual(result["flags"].get("short_answer"), 1)

    def test_repeat_ratio_is_bounded(self):
        self.assertGreater(pack._repeat_ratio("one two three " * 20), 0.30)
        self.assertEqual(pack._repeat_ratio("one two"), 0.0)


if __name__ == "__main__":
    unittest.main()
