"""Synthetic fixture tests for the offline plan-state diagnostic."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts import neral_chat_state_diagnostic as diagnostic


EXPECTED = {
    "correct_place": {"place": "museum", "slot": "10:00", "bring": ["notebook"]},
    "clear_slot": {"place": "park", "slot": None, "bring": ["bottle of water", "kite"]},
    "remove_and_readd": {"place": "station", "slot": "19:30", "bring": ["map"]},
    "unknown_place": {"place": None, "slot": "09:45", "bring": []},
    "replace_items": {"place": "pool", "slot": "11:00", "bring": ["swim cap"]},
    "latest_update": {"place": "square", "slot": None, "bring": ["thermos"]},
}


def fixture():
    return {
        case_id: {arm: {"transcript_sha256": item["transcript_sha256"],
                        "answer": json.dumps(EXPECTED[case_id])} for arm in diagnostic.ARMS}
        for case_id, item in diagnostic.prompts().items()
    }


class ChatStateTests(unittest.TestCase):
    def test_preserves_updates_erasure_and_independent_item_order(self):
        bundle = fixture()
        bundle["clear_slot"]["raw_qwen35"]["answer"] = json.dumps(
            {"place": "park", "slot": "15:00", "bring": ["kite", "bottle of water"]}
        )
        bundle["latest_update"]["neral"]["answer"] = json.dumps(
            {"place": "bakery", "slot": None, "bring": ["thermos"]}
        )
        report = diagnostic.score(bundle)
        self.assertEqual(report["correct"], {"raw_qwen35": 5, "neral": 5})
        self.assertEqual((report["neral_wins"], report["neral_losses"], report["ties"]), (1, 1, 4))
        self.assertFalse(report["independent_holdout"])
        self.assertFalse(report["model_identity_verified"])
        self.assertFalse(report["release_validated"])

    def test_rejects_missing_arms_cases_and_different_transcripts(self):
        valid = fixture()
        for case_id in valid:
            self.assertEqual(len(valid[case_id]["neral"]["transcript_sha256"]), 64)
        for change in ("case", "arm", "digest"):
            with self.subTest(change=change), self.assertRaises(diagnostic.DiagnosticError):
                bundle = copy.deepcopy(valid)
                if change == "case":
                    del bundle["clear_slot"]
                elif change == "arm":
                    del bundle["clear_slot"]["neral"]
                else:
                    bundle["clear_slot"]["neral"]["transcript_sha256"] = "0" * 64
                diagnostic.score(bundle)

    def test_does_not_credit_prose_conflicting_fields_or_duplicate_items(self):
        for response in (
            'Answer: {"place":"museum","slot":"10:00","bring":["notebook"]}',
            '{"place":"museum","slot":"10:00","bring":["notebook"],"place":"library"}',
            '{"place":"museum","slot":"10:00","bring":["notebook","notebook"]}',
            '{"place":"museum","slot":"10:00","bring":["notebook"],"note":"extra"}',
        ):
            with self.subTest(response=response):
                bundle = fixture()
                bundle["correct_place"]["neral"]["answer"] = response
                self.assertEqual(diagnostic.score(bundle)["correct"]["neral"], 5)

    def test_cli_lists_only_prompts_and_scores_without_echoing_observations(self):
        with patch("sys.argv", ["diagnostic", "prompts"]), redirect_stdout(io.StringIO()) as stdout:
            diagnostic.main()
        payload = json.loads(stdout.getvalue())
        for row in payload.values():
            self.assertEqual(set(row), {"messages", "transcript_sha256"})
        self.assertEqual(len(payload), 6)
        stdin = io.TextIOWrapper(io.BytesIO(json.dumps(fixture()).encode()))
        with patch("sys.argv", ["diagnostic", "score"]), patch("sys.stdin", stdin), \
                redirect_stdout(io.StringIO()) as stdout:
            diagnostic.main()
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["correct"], {"raw_qwen35": 6, "neral": 6})
        self.assertNotIn("notebook", stdout.getvalue())
        self.assertNotIn("transcript_sha256", stdout.getvalue())

    def test_cli_fails_closed_on_duplicate_bundle_keys_and_size_limit(self):
        malformed = [b'{"clear_slot":{},"clear_slot":{}}', b" " * 131073]
        for raw in malformed:
            with self.subTest(size=len(raw)), patch("sys.argv", ["diagnostic", "score"]), \
                    patch("sys.stdin", io.TextIOWrapper(io.BytesIO(raw))), \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as exc:
                diagnostic.main()
            self.assertEqual(exc.exception.code, 2)
