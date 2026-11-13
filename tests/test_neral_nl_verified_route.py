"""RED: independent fully parsed source certificates vs NL model consensus.

CPU fixtures only, no Ollama, prospective blind100, or GPU.
"""
import copy
import json
import unittest
from unittest.mock import patch

from neral_nl_verified_route import strict_source_certificate, run_guarded_nl_bridge
from neral_nl_schema_bridge import _parse_candidate
from neral_typed_constraints_v2 import PREFIX, solve_typed_constraints
from scripts.neral_nl_schema_probe import old_case_question


def raw(doc):
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


class SourceCertificateContracts(unittest.TestCase):
    def test_old_public_pin_is_exact_one_using_one_based_positions(self):
        d = strict_source_certificate(old_case_question("21453"))
        self.assertIsInstance(d, dict)
        self.assertEqual(set(d["variables"]), {f"p{i}" for i in range(1, 6)})
        for spec in d["variables"].values():
            self.assertEqual(spec, {"type": "int", "domain": [1, 2, 3, 4, 5]})
        self.assertEqual(d["constraints"][0]["all_different"],
                         ["p1", "p2", "p3", "p4", "p5"])
        self.assertEqual(solve_typed_constraints(PREFIX + raw(d)), 1)
        self.assertEqual(_parse_candidate(raw(d)), d)
        self.assertLess(len(raw(d).split()), 50)

    def test_nonfixture_code_with_same_syntax_is_independently_parsed(self):
        q = ("Ein vierstelliger Code verwendet 1, 2, 3 und 4 genau einmal. "
             "Die 1 steht vor der 4, nicht direkt davor. "
             "Zwischen der 2 und 3 steht genau eine Ziffer. "
             "Die 4 steht unmittelbar rechts von der 2. "
             "Die 3 steht nicht am Rand. Bestimme alle gültigen Codes.")
        d = strict_source_certificate(q)
        self.assertIsInstance(d, dict)
        self.assertEqual(set(d["variables"]), {"p1", "p2", "p3", "p4"})
        self.assertIsInstance(solve_typed_constraints(PREFIX + raw(d)), int)

    def test_historical_graph_and_binary_are_full_source_parsed(self):
        expected = {"logic-02": 6, "logic-06": 4}
        for key, count in expected.items():
            with self.subTest(key=key):
                d = strict_source_certificate(old_case_question(key))
                self.assertIsNotNone(d)
                self.assertEqual(solve_typed_constraints(PREFIX + raw(d)), count)

    def test_arbitrary_additional_conditions_never_get_source_proof(self):
        q = old_case_question("21453")
        for variant in (
            q + " Außerdem steht die 1 links.",
            q.replace("nicht direkt davor", "vielleicht direkt davor"),
            q.replace("Die 3 steht unmittelbar rechts", "Die 3 steht links"),
            q.replace("nicht am Rand", "am Rand"),
            q.replace("genau einmal", "ungefähr einmal"),
            "Count X in 0..3 with X<=2 and X<=1.",
        ):
            self.assertIsNone(strict_source_certificate(variant), variant)

    def test_strict_source_certificate_not_for_multi_turn_or_oversized(self):
        self.assertIsNone(strict_source_certificate("x" * 5000))
        self.assertIsNone(strict_source_certificate(""))
        self.assertIsNone(strict_source_certificate(None))


class GuardedNLLiveRouteContracts(unittest.TestCase):
    def test_both_correct_plus_source_grounding_promotes_exact_count(self):
        q = old_case_question("21453")
        canonical = strict_source_certificate(q)
        calls = []
        def first(turns):
            calls.append(("adapter", turns))
            return raw(canonical)
        def second(turns):
            calls.append(("base", turns))
            return raw(canonical)
        def no_fallback(_):
            self.fail("Independently verified exact result needs no raw Qwen call")
        reply = run_guarded_nl_bridge(
            [{"role": "system", "content": "NERAL"},
             {"role": "user", "content": q}],
            first, second, no_fallback)
        self.assertEqual(reply["text"], "1")
        self.assertEqual(reply["candidate_count"], 1)
        self.assertEqual(reply["consensus_status"], "consensus_unverified")
        self.assertEqual(reply["proof_backend"], "z3_unsat")
        self.assertTrue(reply["verified"])
        self.assertTrue(reply["safe_to_promote"])
        self.assertFalse(reply["model_generated"])
        self.assertEqual(reply["selection_policy"], "independently_source_verified_nl")
        self.assertEqual([x[0] for x in calls], ["adapter", "base"])
        self.assertNotEqual(calls[0][1][0]["content"], calls[1][1][0]["content"])

    def test_two_models_agree_on_wrong_omission_but_cannot_promote(self):
        q = old_case_question("21453")
        canonical = strict_source_certificate(q)
        bad = copy.deepcopy(canonical)
        bad["constraints"].pop(3)
        captured = []
        def fallback(turns):
            captured.append(turns)
            return "BASIS_QWEN"
        reply = run_guarded_nl_bridge(
            [{"role": "system", "content": "NERAL"}, {"role": "user", "content": q}],
            lambda _: raw(bad), lambda _: raw(bad), fallback)
        self.assertEqual(reply["text"], "BASIS_QWEN")
        self.assertFalse(reply["verified"])
        self.assertFalse(reply["safe_to_promote"])
        self.assertEqual(reply["response_source"], "raw_qwen")
        self.assertEqual(len(captured), 1)

    def test_unknown_prose_skips_model_proposals_and_stays_base(self):
        message = [{"role": "system", "content": "NERAL"},
                   {"role": "user", "content": "What is a pulsar?"}]
        observed = []
        def base(turns):
            observed.extend(turns)
            return "BASE_QWEN"
        reply = run_guarded_nl_bridge(
            message, lambda _: self.fail("no proposal"),
            lambda _: self.fail("no proposal"), base)
        self.assertEqual(reply["text"], "BASE_QWEN")
        self.assertFalse(reply["safe_to_promote"])
        self.assertEqual(observed, message)

    def test_same_wrong_count_and_lexical_coverage_not_enough(self):
        # Both can agree numerically but omit a source constraint.
        q = old_case_question("logic-06")
        d = strict_source_certificate(q)
        bad = copy.deepcopy(d)
        bad["constraints"].pop()
        reply = run_guarded_nl_bridge(
            [{"role":"system","content":"NERAL"},{"role":"user","content":q}],
            lambda _: raw(bad),lambda _: raw(bad), lambda _: "RAW_BASE")
        self.assertFalse(reply["safe_to_promote"])
        self.assertEqual(reply["text"], "RAW_BASE")

    def test_mismatched_ast_and_failed_z3_never_override_raw(self):
        q = old_case_question("logic-02")
        d = strict_source_certificate(q)
        with patch("neral_nl_verified_route._equivalence", return_value=(False,"z3_unknown")):
            reply = run_guarded_nl_bridge(
                [{"role":"system","content":"NERAL"},{"role":"user","content":q}],
                lambda _: raw(d),lambda _: raw(d),lambda _: "BASE")
        self.assertEqual(reply["text"],"BASE")
        self.assertFalse(reply["verified"])

    def test_no_multi_turn_nl_override_even_if_fullmatch_on_last_user(self):
        q = old_case_question("21453")
        turns = [{"role":"system","content":"NERAL"},
                 {"role":"user","content":"Earlier question"},
                 {"role":"assistant","content":"Answer"},
                 {"role":"user","content":q}]
        reply = run_guarded_nl_bridge(
            turns, lambda _: self.fail("not isolated"),
            lambda _: self.fail("not isolated"),
            lambda x: "RAW")
        self.assertEqual(reply["text"],"RAW")
        self.assertFalse(reply["safe_to_promote"])

    def test_existing_pr537_never_accepts_plain_model_consensus(self):
        from neral_tir_regression_guard import choose_guarded_tir_reply
        q = old_case_question("logic-02")
        d = strict_source_certificate(q)
        result = choose_guarded_tir_reply(
            {"text":"6","mode":"HIGH","verified":False,
             "model_generated":True,"verification_status":"consensus_unverified",
             "verification_scope":"nl_model_consensus","tool_used":"neral_logic_exact",
             "tool_calls_succeeded":1,"calls_used":2,"seconds":1.0},
            [{"role":"system","content":"NERAL"},{"role":"user","content":q}],
            lambda _: "UNCHANGED_BASE")
        self.assertEqual(result["text"],"UNCHANGED_BASE")
        self.assertFalse(result["verified"])


if __name__ == "__main__":
    unittest.main()
