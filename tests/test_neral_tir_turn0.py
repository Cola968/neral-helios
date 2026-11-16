"""RED: Turn-0 constrained JSON extraction, deterministic fail-closed fallback.

CPU-only; no Ollama GPU, network, cloud, generated code or personal data.
"""
from __future__ import annotations
import json
import copy
import unittest
from unittest.mock import patch
from jsonschema import Draft202012Validator

from neral_turn0_extractor import EXTRACT_PROMPT, parse_extracted_constraints
from neral_tir_experimental import run_experimental_tir

QUESTION = ("A and B each choose a number from 1 to 3. "
            "A must be smaller than B. How many assignments are possible?")
SUFFIX = "\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."
SCHEMA = {
    "variables": {"A": [1, 2, 3], "B": [1, 2, 3]},
    "rules": [{"left": "A", "op": "<", "right": "B"}],
    "query": "count",
}
EVIDENCE = ["each choose a number from 1 to 3", "A must be smaller than B",
            "How many assignments are possible?"]

def packet(*, kind="formal", complete=True, schema=SCHEMA, evidence=EVIDENCE):
    return json.dumps({
        "kind": kind, "complete": complete, "schema": schema,
        "evidence": evidence,
    }, separators=(",", ":"))

def messages(question):
    return [{"role": "system", "content": "Du bist NERAL."},
            {"role": "user", "content": question}]


def model_trace_messages(question):
    """Exercise the model-controlled tool review, not the direct one-turn path."""
    return [messages(question)[0],
            {"role": "user", "content": "Earlier unrelated chat."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": question}]


class ExtractionSchemaContracts(unittest.TestCase):
    def test_one_shot_per_logic_family_and_strict_json_without_xml(self):
        self.assertEqual(EXTRACT_PROMPT.count("Example input:"), 2)
        self.assertIn("JSON", EXTRACT_PROMPT)
        self.assertNotIn("<tool_call", EXTRACT_PROMPT)
        self.assertIn('"bool_rules":', EXTRACT_PROMPT)
        self.assertIn('"op":"xor"', EXTRACT_PROMPT)
        self.assertIn("first output character", EXTRACT_PROMPT)

    def test_boolean_few_shot_round_trips_into_exact_solver(self):
        question = ("A and B can each be True or False. "
                    "A XOR B must be True. "
                    "How many assignments satisfy all conditions?")
        candidate = {
            "kind": "formal", "complete": True,
            "schema": {"variables": {"A": [False, True],
                                      "B": [False, True]},
                       "rules": [], "bool_rules": [
                           {"op": "eq", "args": [
                               {"op": "xor", "args": ["A", "B"]}, True]}],
                       "query": "count"},
            "evidence": ["A and B can each be True or False",
                         "A XOR B must be True",
                         "How many assignments satisfy all conditions?"],
        }
        self.assertEqual(parse_extracted_constraints(
            question, json.dumps(candidate, separators=(",", ":"))), 2)

    def test_valid_model_extraction_is_a_computed_but_unverified_answer(self):
        self.assertEqual(parse_extracted_constraints(QUESTION, packet()), 3)

    def test_rejects_unproven_incomplete_or_oversized_proposals(self):
        from neral_formal_constraints import MAX_ASSIGNMENTS
        self.assertEqual(MAX_ASSIGNMENTS, 100000)
        variants = [
            "", "not JSON", "{bad", " "+packet(), packet()+"\nhello",
            '{"kind":"none"}'+packet(),
            packet(complete=False), packet(evidence=[]),
            packet(evidence=["not in source"]*3),
            packet(evidence=EVIDENCE[:1]), packet(kind="unknown"),
            json.dumps({"kind":"formal","complete":True,"schema":SCHEMA,
                        "evidence":EVIDENCE,"code":"print(1)"}),
            packet(schema={**SCHEMA,"variables":{
                "A":list(range(12)),"B":list(range(12)),
                "C":list(range(12)),"D":list(range(12)),
                "E":list(range(12))}}),
            '{"kind":"formal","kind":"none"}',
            packet(schema={**SCHEMA,"rules":[]}),
        ]
        for value in variants:
            with self.subTest(value=str(value)[:55]):
                self.assertIsNone(parse_extracted_constraints(QUESTION,value))
        self.assertIsNone(parse_extracted_constraints(QUESTION,'{"kind":"none"}'))

    def test_natural_language_is_never_verified_by_model_self_certification(self):
        replies=iter(["Fallback untouched."])
        observed=[]
        def respond(turns):
            observed.append([dict(row) for row in turns])
            return next(replies)
        extract_calls=[]
        def extract(turns):
            extract_calls.append(turns)
            return packet()
        report=run_experimental_tir(
            messages(QUESTION+SUFFIX),respond,extract_constraints=extract,
            first_turn_diagnostic=lambda:{
                "eval_count":94,"prompt_eval_count":122,"done_reason":"stop"})
        self.assertEqual(len(extract_calls),1)
        self.assertEqual(len(observed),0)
        self.assertEqual(report["text"],"FINAL_ANSWER: 3")
        self.assertEqual(report["verification_status"],"model_extracted_schema_unverified")
        self.assertFalse(report["verified"])
        self.assertTrue(report["model_generated"])
        self.assertEqual(report["verification_scope"],"model_extracted_finite_domain")
        self.assertEqual(report["tool_used"],"neral_logic_exact")
        self.assertEqual(report["calls_used"],1)
        self.assertEqual(report["tool_calls_succeeded"],1)
        self.assertEqual(report["first_turn_telemetry"]["eval_count"],94)
        self.assertEqual(report["first_turn_telemetry"]["decision"],"extracted_formal")

    def test_nonformal_refusal_keeps_tool_decision_model_controlled(self):
        original=messages("How does rain form?")
        observed=[]
        def respond(turns):
            observed.append([dict(row) for row in turns])
            return "NERAL_FINAL_TEXT: Rain forms after condensation."
        report=run_experimental_tir(original,respond,
                extract_constraints=lambda _: '{"kind":"none"}')
        self.assertEqual(len(observed),1)
        self.assertNotEqual(observed[0],original)
        self.assertEqual(report["text"],"Rain forms after condensation.")
        self.assertEqual(report["calls_used"],2)
        self.assertEqual(report["tool_calls_succeeded"],0)
        self.assertEqual(report["first_turn_telemetry"]["decision"],"extraction_none")
        self.assertEqual(report["first_turn_telemetry"]["extraction_rejection_reason"],"kind_none")

    def test_malformed_or_semantically_unsupported_falls_back_without_executing_code(self):
        for proposed in ("not JSON", packet(complete=False),
                         packet(evidence=["invented"]*3)):
            with self.subTest(proposed=proposed[:20]):
                original=messages(QUESTION)
                seen=[]
                result=run_experimental_tir(original,
                    lambda m:(seen.append(m) or "Raw safe answer"),
                    extract_constraints=lambda _:proposed,
                    tool_runner=lambda *_:self.fail("no generated Python"))
                self.assertEqual(seen,[original])
                self.assertEqual(result["text"],"Raw safe answer")
                self.assertFalse(result["verified"])
                self.assertEqual(result["tool_calls_triggered"],0)

    def test_extractor_timeout_or_length_causes_direct_fallback(self):
        original=messages(QUESTION)
        def fail(_):
            raise RuntimeError("local model truncated")
        report=run_experimental_tir(original, lambda _: "Raw fallback",
                                    extract_constraints=fail)
        self.assertEqual(report["text"],"Raw fallback")
        self.assertEqual(report["calls_used"],2)

    def test_legacy_tir_without_extractor_is_unchanged(self):
        seen=[]
        def respond(turns):
            seen.append([dict(row) for row in turns])
            return "NERAL_FINAL_TEXT: ordinary answer"
        result=run_experimental_tir(messages("Explain a nebula"), respond)
        self.assertEqual(result["text"],"ordinary answer")
        self.assertEqual(len(seen),1)
        self.assertNotIn("extraction_status",result)


class OllamaExtractWireContracts(unittest.TestCase):
    def extraction_format(self, question):
        from neral_agent_autonomy import OllamaLocalBackend
        requests = []
        class Local:
            def open(self, request, timeout):
                requests.append(json.loads(request.data))
                class Response:
                    def __enter__(self): return self
                    def __exit__(self, *args): return False
                    def read(self, *args):
                        return json.dumps({"done": True, "done_reason": "stop",
                            "message": {"content": '{"kind":"none"}'}}).encode()
                return Response()
        backend = OllamaLocalBackend("qwen3.5:9b", timeout=90,
                                    inference_profile="measured_nonthinking_2048")
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            backend.respond_formal_extract(messages(question))
        schema = requests[0]["format"]
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)

    def test_prerecognized_logic_cannot_emit_none_or_empty_constraints(self):
        questions = [
            "P and Q are Boolean variables. P XOR Q is true. Count assignments.",
            "P und Q sind boolesche Variablen. Genau eine ist wahr. Wie viele Belegungen?",
            "Put Ada, Ben, Cy in a complete order, each exactly once. Ada precedes Ben. Count orders.",
            "Ada, Ben, Cy stehen in einer Reihenfolge, jede Person genau einmal. Ada vor Ben. Wie viele Anordnungen?",
        ]
        for question in questions:
            validator = self.extraction_format(question)
            for candidate in ({"kind": "none"}, {"kind": "formal"},
                              {"kind": "formal", "complete": True, "schema": {},
                               "evidence": ["source domain", "source query"]}):
                with self.subTest(question=question, candidate=candidate):
                    self.assertFalse(validator.is_valid(candidate))

    def test_wire_schema_rejects_missing_fields_extras_bad_ops_and_empty_domains(self):
        validator = self.extraction_format(QUESTION)
        valid = json.loads(packet())
        valid["schema"]["variables"] = [
            {"name": "A", "domain": [1, 2, 3]},
            {"name": "B", "domain": [1, 2, 3]},
        ]
        self.assertTrue(validator.is_valid(valid))
        invalid = []
        for key in valid:
            row = copy.deepcopy(valid)
            del row[key]
            invalid.append(row)
        for change in ({"complete": False}, {"code": "print(1)"},
                       {"evidence": []}, {"evidence": ["only one span"]}):
            invalid.append({**valid, **change})
        for change in ({"variables": {}}, {"variables": []},
                       {"variables": [{"name": "A", "domain": []}]},
                       {"variables": [{"name": "A", "domain": [1], "code": "eval"}]},
                       {"rules": []}, {"query": "explain"}, {"code": "eval"},
                       {"rules": [{"left": "A", "op": "exec", "right": "B"}]},
                       {"rules": [{"left": "A", "op": "<", "right": "B", "value": 1}]}):
            invalid.append({**valid, "schema": {**valid["schema"], **change}})
        for row in invalid:
            with self.subTest(candidate=row):
                self.assertFalse(validator.is_valid(row))

    def test_variable_list_normalizes_without_losing_domains_or_allowing_duplicates(self):
        candidate = json.loads(packet())
        variables = [{"name": "A", "domain": [1, 2, 3]},
                     {"name": "B", "domain": [1, 2, 3]}]
        candidate["schema"]["variables"] = variables
        self.assertEqual(parse_extracted_constraints(QUESTION, json.dumps(candidate)), 3)
        for invalid in ([], variables + [variables[0]],
                        [{"name": "A", "domain": [1], "code": "eval"}],
                        [{"name": [], "domain": [1]}], ["A"]):
            candidate["schema"]["variables"] = invalid
            self.assertIsNone(parse_extracted_constraints(QUESTION, json.dumps(candidate)))

    def test_actual_prompt_examples_are_source_grounded_solvable_and_wire_valid(self):
        # Parse the literal examples the model actually receives, not a copy.
        examples = EXTRACT_PROMPT.split("Example input: ")[1:]
        self.assertEqual(len(examples), 2)
        for section, expected in zip(examples, (2, 3)):
            question, encoded = section.split("Example JSON: ", 1)
            candidate, _ = json.JSONDecoder().raw_decode(encoded)
            self.assertTrue(self.extraction_format(question).is_valid(candidate))
            self.assertEqual(parse_extracted_constraints(
                question, json.dumps(candidate, separators=(",", ":"))), expected)

    def test_nonconstraint_question_can_still_abstain(self):
        self.assertTrue(self.extraction_format("Explain rain.").is_valid({"kind": "none"}))

    def test_boolean_arity_depth_and_ordering_bounds_are_closed_on_the_wire(self):
        question = ("P and Q are Boolean variables. P XOR Q is true. Count assignments.")
        validator = self.extraction_format(question)
        candidate = {"kind": "formal", "complete": True, "schema": {
            "variables": [{"name": "P", "domain": [False, True]},
                          {"name": "Q", "domain": [False, True]}],
            "rules": [], "bool_rules": [{"op": "xor", "args": ["P", "Q"]}],
            "query": "count"}, "evidence": ["Boolean variables", "Count assignments"]}
        self.assertTrue(validator.is_valid(candidate))
        nested = "P"
        for _ in range(5):
            nested = {"op": "not", "args": [nested]}
        for rule in ({"op": "not", "args": ["P", "Q"]},
                     {"op": "eq", "args": ["P"]},
                     {"op": "xor", "args": []},
                     {"op": "exec", "args": ["P"]},
                     {"op": "at_least", "args": ["P", "Q"]}, nested):
            candidate["schema"]["bool_rules"] = [rule]
            self.assertFalse(validator.is_valid(candidate))
        candidate["schema"] = {"ordering": ["Ada", "Ben"], "rules": [
            {"left": "Ada", "op": "<", "right": "Ben"}], "query": "count"}
        self.assertTrue(validator.is_valid(candidate))
        for items in ([], ["Ada"], list("ABCDEFGHI")):
            candidate["schema"]["ordering"] = items
            self.assertFalse(validator.is_valid(candidate))

    def test_all_generated_reasoning_cases_remove_the_abstention_branch(self):
        from scripts.neral_high12_manual import select_cases
        for case in select_cases(123456):
            if case["domain"] == "reasoning":
                with self.subTest(family=case["family"]):
                    self.assertFalse(self.extraction_format(case["prompt"]).is_valid({"kind":"none"}))

    def test_only_extractor_uses_json_384_temp0_and_raw_remains_2048(self):
        from neral_agent_autonomy import OllamaLocalBackend
        requests=[]
        timeouts=[]
        class Local:
            def open(self,request,timeout):
                requests.append(json.loads(request.data))
                timeouts.append(timeout)
                class Response:
                    def __enter__(self): return self
                    def __exit__(self,*args): return False
                    def read(self,*args):
                        return json.dumps({"done":True,"done_reason":"stop",
                            "eval_count":64,"message":{"content":"{\"kind\":\"none\"}"}}).encode()
                return Response()
        b=OllamaLocalBackend("qwen3.5:9b", timeout=30,
                            inference_profile="measured_nonthinking_2048")
        turns=messages("Explain rain.")
        with patch("neral_agent_autonomy.local_only_opener",return_value=Local()):
            b.respond_formal_extract(turns)
            b.respond_chat(turns)
        self.assertEqual(len(requests),2)
        self.assertIsInstance(requests[0]["format"], dict)
        self.assertTrue(Draft202012Validator(requests[0]["format"]).is_valid({"kind":"none"}))
        self.assertNotIn("format", requests[1])
        self.assertFalse(requests[0]["think"])
        self.assertEqual(requests[0]["options"]["num_predict"],384)
        self.assertEqual(requests[0]["options"]["temperature"],0)
        self.assertEqual(requests[1]["options"]["num_predict"],2048)
        self.assertEqual(timeouts, [30, 90])
        self.assertNotIn("format",requests[1])

if __name__ == "__main__":
    unittest.main()


class Turn0RouterDiagnosticRegression(unittest.TestCase):
    def test_explicit_none_continues_to_model_driven_python_tool_decision(self):
        # Not a benchmark case: proves the router works for arbitrary code.
        original = messages("What does a local Python sum of 6 and 7 produce?")
        decisions = []
        replies = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(sum([6,7]))"}',
            "NERAL_FINAL_TEXT: 13",
        ])
        def model(turns):
            decisions.append([dict(row) for row in turns])
            return next(replies)
        executed = []
        def tool(tool, code):
            executed.append((tool, code))
            return {"status": "PASSED", "output": "13",
                    "verification_scope": "execution_only"}
        result = run_experimental_tir(
            original, model, extract_constraints=lambda _: '{"kind":"none"}',
            tool_runner=tool, first_turn_diagnostic=lambda: {
                "eval_count": 6, "done_reason": "stop", "prompt_eval_count": 500})
        self.assertEqual(result["text"], "13")
        self.assertEqual(result["tool_calls_triggered"], 1)
        self.assertEqual(result["tool_calls_succeeded"], 1)
        self.assertEqual(result["calls_used"], 3)
        self.assertEqual(executed, [("python_sandbox", "print(sum([6,7]))")])
        self.assertIn("NERAL_EXECUTE_JSON:", decisions[0][0]["content"])
        self.assertEqual(result["first_turn_telemetry"]["decision"], "extraction_none")

    def test_parser_reports_safe_rejection_enum_without_disclosing_text(self):
        errors = []
        result = parse_extracted_constraints(QUESTION, packet(evidence=["unquoted"] * 3),
                                             rejection_reason_sink=errors)
        self.assertIsNone(result)
        self.assertEqual(errors, ["evidence_quote_invalid"])
        self.assertNotIn("unquoted", errors[0])
        errors = []
        self.assertIsNone(parse_extracted_constraints(
            QUESTION, '{"kind":"none"}', rejection_reason_sink=errors))
        self.assertEqual(errors, ["kind_none"])

    def test_invalid_formal_stays_closed_and_never_executes_fallback_tool(self):
        question = "How many valid assignments exist for A and B?"
        observed=[]
        out=run_experimental_tir(
            messages(question),
            lambda turns: (observed.append([dict(t) for t in turns]) or
                           "Not enough information."),
            extract_constraints=lambda _: '{"kind":"formal","complete":true}',
            tool_runner=lambda *_:self.fail("Invalid proposals must not execute"))
        self.assertEqual(out["tool_calls_succeeded"],0)
        self.assertEqual(out["verification_status"],"base_fallback_unverified")
        self.assertEqual(len(observed),1)

class EvidenceFormatAndToolDirectiveContracts(unittest.TestCase):
    """Synthetic CPU cases: the private GPU JSON was never retained."""

    def test_source_quotes_allow_whitespace_and_terminal_punctuation_variations(self):
        variants = [
            "each   choose a number from 1 to 3",
            "A must be smaller than B!",
            "How many assignments are possible",
        ]
        rejections = []
        self.assertEqual(parse_extracted_constraints(
            QUESTION, packet(evidence=variants),
            rejection_reason_sink=rejections), 3)
        self.assertEqual(rejections, ["accepted"])

    def test_three_conditions_can_share_one_source_grounded_quote(self):
        question = ("A, B, C are Boolean variables. "
                    "A XOR B must be True; B OR C must be True; "
                    "A OR C must be True. "
                    "How many assignments satisfy all conditions?")
        tree = lambda op, x, y: {
            "op": "eq",
            "args": [{"op": op, "args": [x, y]}, True],
        }
        candidate = {
            "kind": "formal", "complete": True,
            "schema": {
                "variables": {"A": [False, True], "B": [False, True],
                              "C": [False, True]},
                "rules": [], "bool_rules": [
                    tree("xor", "A", "B"), tree("or", "B", "C"),
                    tree("or", "A", "C"),
                ], "query": "count",
            },
            "evidence": [
                "A, B, C are Boolean variables",
                "A XOR B must be True; B OR C must be True; A OR C must be True",
                "How many assignments satisfy all conditions?",
            ],
        }
        reasons = []
        self.assertEqual(parse_extracted_constraints(
            question, json.dumps(candidate, separators=(",", ":")),
            rejection_reason_sink=reasons), 2)
        self.assertEqual(reasons, ["accepted"])

    def test_different_meaning_or_unquoted_paraphrase_is_still_rejected(self):
        for evidence in (
            ["each select a number between 1 and 3",
             "A must be less than B", "How many assignments are possible?"],
            ["each choose a number from 1 to 3",
             "A must NOT be smaller than B", "How many assignments are possible?"],
        ):
            with self.subTest(evidence=evidence):
                reasons = []
                self.assertIsNone(parse_extracted_constraints(
                    QUESTION, packet(evidence=evidence),
                    rejection_reason_sink=reasons))
                self.assertEqual(reasons, ["evidence_quote_invalid"])

    def test_normalized_duplicate_and_missing_quotes_report_distinct_codes(self):
        cases = (
            (["A must be smaller than B",
              "A must be smaller than B.",
              "How many assignments are possible?"], "evidence_duplicate"),
            (EVIDENCE[:1], "evidence_count_invalid"),
            (["tiny", "A must be smaller than B",
              "How many assignments are possible?"], "evidence_quote_invalid"),
        )
        for evidence, expected in cases:
            with self.subTest(expected=expected):
                reasons = []
                self.assertIsNone(parse_extracted_constraints(
                    QUESTION, packet(evidence=evidence),
                    rejection_reason_sink=reasons))
                self.assertEqual(reasons, [expected])

    def test_none_handoff_explicitly_requires_python_sandbox_for_code_tasks(self):
        original = messages(
            "What exact integer does this local Python program print?\\n"
            "data=[2,3]\\nprint(sum(data))")
        observed = []
        replies = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(sum([2,3]))"}',
            "NERAL_FINAL_TEXT: 5",
        ])
        def respond(turns):
            observed.append([dict(x) for x in turns])
            return next(replies)
        executed = []
        def tool(tool_name, code):
            executed.append((tool_name, code))
            return {"status": "PASSED", "output": "5",
                    "verification_scope": "execution_only"}
        result = run_experimental_tir(
            original, respond, extract_constraints=lambda _: '{"kind":"none"}',
            tool_runner=tool)
        self.assertEqual(result["text"], "5")
        self.assertEqual(result["tool_calls_succeeded"], 1)
        self.assertEqual(executed, [("python_sandbox", "print(sum([2,3]))")])
        directive = observed[0][0]["content"]
        self.assertIn("CONSTRAINT EXTRACTOR ABSTAINED", directive)
        self.assertIn("python_sandbox", directive)
        self.assertIn("MUST", directive)
        self.assertIn("before", directive.lower())
        self.assertNotIn("CONSTRAINT EXTRACTOR ABSTAINED", original[0]["content"])


class SolverRejectionAndToolProvenanceCPU(unittest.TestCase):
    """CPU-only replay of safe synthetic failures; no raw GPU model text exists."""

    def test_successful_python_tool_name_survives_unverified_final_fallback(self):
        from scripts.neral_same_base_live import checked_high_packet
        responses = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(sum([2,3]))"}',
            "NERAL_FINAL_TEXT: UNVERIFIED",
            "FINAL_ANSWER: 5",
        ])
        runs = []
        def model(turns):
            runs.append([dict(x) for x in turns])
            return next(responses)
        result = run_experimental_tir(
            messages("Calculate output of this Python program: print(sum([2,3]))"),
            model, extract_constraints=lambda _: '{"kind":"none"}',
            tool_runner=lambda tool, code: {
                "status": "PASSED", "output": "5",
                "verification_scope": "execution_only",
            })
        self.assertEqual(len(runs), 3)
        self.assertEqual(result["text"], "FINAL_ANSWER: 5")
        self.assertEqual(result["tool_calls_triggered"], 1)
        self.assertEqual(result["tool_calls_succeeded"], 1)
        self.assertEqual(result.get("tool_used"), "neral_python_sandbox")
        self.assertEqual(result["verification_status"], "base_fallback_unverified")
        self.assertIs(result["verified"], False)
        self.assertIs(result.get("tool_output_reviewed"), False)
        self.assertEqual(result["verification_scope"], "none")
        self.assertEqual(checked_high_packet({
            **result, "model": "qwen3.5:9b",
            "inference_profile": "measured_nonthinking_2048",
            "model_origin": "local_foundation_not_neral_adapter",
            "adapter_loaded": False,
        }), "FINAL_ANSWER: 5")

    def test_failed_tool_has_no_successful_tool_name(self):
        responses = iter([
            'NERAL_EXECUTE_JSON: {"tool":"solve","code":"1/0"}',
            "FINAL_ANSWER: 3",
        ])
        result = run_experimental_tir(
            messages("Compute 1/0 and explain"), lambda _: next(responses),
            tool_runner=lambda *_: {"status": "FAILED", "output": ""})
        self.assertEqual(result["tool_calls_triggered"], 1)
        self.assertEqual(result["tool_calls_succeeded"], 0)
        self.assertIsNone(result.get("tool_used"))
        self.assertIs(result.get("tool_output_reviewed"), False)

    def test_checker_rejects_forged_fallback_tool_provenance(self):
        from scripts.neral_same_base_live import checked_high_packet, PilotBlocked
        packet_ = {
            "model": "qwen3.5:9b", "inference_profile": "measured_nonthinking_2048",
            "model_origin": "local_foundation_not_neral_adapter",
            "adapter_loaded": False, "mode": "HIGH",
            "text": "FINAL_ANSWER: 5", "model_generated": True,
            "verified": False, "verification_status": "base_fallback_unverified",
            "verification_scope": "none",
            "tool_used": "neral_python_sandbox",
            "tool_output_reviewed": False,
            "tool_calls_triggered": 1, "tool_calls_succeeded": 1,
        }
        self.assertEqual(checked_high_packet(packet_), "FINAL_ANSWER: 5")
        for change in [
            {"tool_calls_succeeded": 0},
            {"tool_calls_triggered": 0},
            {"tool_output_reviewed": True},
            {"verified": True},
            {"verification_scope": "numeric_final_only"},
            {"tool_used": "shell"},
        ]:
            with self.subTest(change=change), self.assertRaises(PilotBlocked):
                checked_high_packet({**packet_, **change})

    def test_solver_unsupported_boolean_operator_diagnosed_without_text_leak(self):
        from copy import deepcopy
        question = ("A and B are Boolean variables. "
                    "A XOR B must be True. "
                    "How many assignments satisfy all conditions?")
        from neral_turn0_extractor import parse_extracted_constraints
        formal = {
            "kind": "formal", "complete": True,
            "schema": {
                "variables": {"A": [False, True], "B": [False, True]},
                "rules": [], "bool_rules": [
                    {"op": "nand", "args": ["A", "B"]}],
                "query": "count",
            },
            "evidence": ["A and B are Boolean variables",
                         "A XOR B must be True",
                         "How many assignments satisfy all conditions?"],
        }
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            question, json.dumps(formal), rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_unsupported_constraint"])
        self.assertEqual(details, ["boolean_operator"])
        self.assertNotIn("nand", str(reasons))

    def test_solver_malformed_relation_is_syntax_error(self):
        broken = {**SCHEMA, "rules": [{"left": "A", "right": "B"}]}
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            QUESTION, packet(schema=broken), rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_syntax_error"])
        self.assertEqual(details, ["malformed_relation"])

    def test_solver_comparison_type_mismatch_is_type_error(self):
        mixed = {**SCHEMA, "variables": {"A": [1, 2], "B": [False, True]}}
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            QUESTION, packet(schema=mixed), rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_type_error"])
        self.assertEqual(details, ["comparison_types"])

    def test_solver_search_budget_rejection_has_safe_bounds_code(self):
        question = ("A B C D E F each can be 0 through 11. "
                    "A must be smaller than B. Count valid assignments.")
        schema = {
            "variables": {name: list(range(12)) for name in "ABCDEF"},
            "rules": [{"left": "A", "op": "<", "right": "B"}],
            "query": "count",
        }
        doc = json.dumps({
            "kind": "formal", "complete": True, "schema": schema,
            "evidence": ["A B C D E F each can be 0 through 11",
                         "A must be smaller than B",
                         "Count valid assignments."],
        })
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            question, doc, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_bounds_exceeded"])
        self.assertEqual(details, ["search_space_hard_cap"])

    def test_unsatisfiable_is_a_valid_zero_not_a_solver_exception(self):
        schema = {
            "variables": {"A": [0, 1], "B": [0, 1]},
            "rules": [
                {"left": "A", "op": "==", "value": 1},
                {"left": "A", "op": "!=", "value": 1},
            ], "query": "count",
        }
        reasons, details = [], []
        self.assertEqual(parse_extracted_constraints(
            QUESTION, packet(schema=schema), rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details), 0)
        self.assertEqual(reasons, ["accepted"])
        self.assertEqual(details, [])

    def test_unknown_solver_error_message_is_not_disclosed(self):
        from neral_formal_constraints import InvalidFormalConstraints
        with patch("neral_turn0_extractor.try_solve_formal_constraints",
                   side_effect=InvalidFormalConstraints("secret private context")):
            reasons = []
            self.assertIsNone(parse_extracted_constraints(
                QUESTION, packet(), rejection_reason_sink=reasons))
        self.assertEqual(reasons, ["solver_rejected"])
        self.assertNotIn("secret", str(reasons))

    def test_solver_specific_reason_codes_remain_scalar_only_through_tir(self):
        schema = {**SCHEMA, "rules": [{"left": "A", "right": "B"}]}
        responses = []
        def raw(turns):
            responses.append(turns)
            return "FINAL_ANSWER: 3"
        event, solver_detail = [], []
        report = run_experimental_tir(
            messages(QUESTION), raw,
            extract_constraints=lambda _: packet(schema=schema),
            extraction_rejection_sink=event,
            solver_rejection_detail_sink=solver_detail,
            first_turn_diagnostic=lambda: {
                "eval_count": 374, "prompt_eval_count": 839, "done_reason": "stop",
            })
        self.assertEqual(report["first_turn_telemetry"]["extraction_rejection_reason"],
                         "solver_syntax_error")
        self.assertEqual(event, ["solver_syntax_error"])
        self.assertEqual(solver_detail, ["malformed_relation"])
        self.assertEqual(report["first_turn_telemetry"]["solver_rejection_detail"],
                         "malformed_relation")
        from scripts.neral_tir_hard_live import safe_first_turn_event
        public = safe_first_turn_event(report["first_turn_telemetry"])
        self.assertEqual(public["solver_rejection_detail"], "malformed_relation")
        with self.subTest("Do not accept arbitrary model detail strings"):
            self.assertIsNone(safe_first_turn_event({
                **report["first_turn_telemetry"],
                "solver_rejection_detail": "private generated code",
            }))
        self.assertEqual(report["tool_calls_succeeded"], 0)
        self.assertEqual(len(responses), 1)


    def test_solver_reason_map_is_exhaustive_for_static_solver_exceptions(self):
        """Future solver validation must not silently collapse known reasons."""
        import ast
        import inspect
        import neral_formal_constraints
        from neral_turn0_extractor import _SAFE_SOLVER_REJECTIONS
        tree = ast.parse(inspect.getsource(neral_formal_constraints))
        fixed = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            call = node.exc
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "InvalidFormalConstraints"
                    and len(call.args) == 1
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)):
                fixed.add(call.args[0].value)
        self.assertEqual(set(_SAFE_SOLVER_REJECTIONS), fixed)
        self.assertEqual(len(fixed), 28)


class BooleanSourceInferenceAndExactCodeExecution(unittest.TestCase):
    """New CPU-only regressions; the 374-token GPU schema is unavailable."""

    @staticmethod
    def boolean_question():
        return ("How many Boolean assignments to A,B,C,D,E,F,G satisfy ALL: "
                "(A XOR C XOR F) is True; (B OR D) is True; E = NOT F; "
                "(A AND G) is False; C implies G; "
                "at least 1 of the seven variables are True? Return one integer.")

    @staticmethod
    def boolean_schema():
        # Mirrors the general Boolean AST shape, NOT recovered GPU output.
        return {
            "variables": {name: [False, True] for name in "ABCDEFG"},
            "rules": [],
            "bool_rules": [
                {"op": "eq", "args": [
                    {"op": "xor", "args": ["A", "C", "F"]}, True]},
                {"op": "or", "args": ["B", "D"]},
                {"op": "eq", "args": ["E", {"op": "not", "args": ["F"]}]},
                {"op": "not", "args": [{"op": "and", "args": ["A", "G"]}]},
                {"op": "implies", "args": ["C", "G"]},
                {"op": "at_least", "args": list("ABCDEFG"), "min": 1},
            ],
            "query": "count",
        }

    def logic_evidence(self):
        q = self.boolean_question()
        return [q[:min(125, q.index(":")+1)],
                "(A XOR C XOR F) is True",
                "(B OR D) is True",
                "E = NOT F",
                "(A AND G) is False",
                "C implies G",
                "at least 1 of the seven variables are True?",
                "Return one integer."]

    def test_infer_binary_integer_domain_only_for_source_declared_bool_vars(self):
        schema = self.boolean_schema()
        from copy import deepcopy
        trial = deepcopy(schema)
        trial["variables"]["C"] = [0, 1]
        trial["variables"]["G"] = [1, 0]
        reason, details = [], []
        actual = parse_extracted_constraints(
            self.boolean_question(), packet(schema=trial, evidence=self.logic_evidence()),
            rejection_reason_sink=reason, solver_rejection_detail_sink=details)
        baseline = parse_extracted_constraints(
            self.boolean_question(), packet(schema=schema, evidence=self.logic_evidence()))
        self.assertIsInstance(actual, int)
        self.assertEqual(actual, baseline)
        self.assertEqual(reason, ["accepted"])
        self.assertEqual(details, [])

    def test_infer_missing_boolean_declaration_only_when_explicitly_named_in_source(self):
        from copy import deepcopy
        schema = self.boolean_schema()
        expected = parse_extracted_constraints(
            self.boolean_question(), packet(schema=schema, evidence=self.logic_evidence()))
        trial = deepcopy(schema)
        del trial["variables"]["G"]
        reasons = []
        actual = parse_extracted_constraints(
            self.boolean_question(), packet(schema=trial, evidence=self.logic_evidence()),
            rejection_reason_sink=reasons)
        self.assertEqual(actual, expected)
        self.assertEqual(reasons, ["accepted"])

    def test_bool_domain_inference_rejects_nonbinary_and_case_conflicts(self):
        from copy import deepcopy
        question = self.boolean_question()
        evidence = self.logic_evidence()
        schema = self.boolean_schema()
        for change in (
            lambda t: t["variables"].update(C=[0, 2]),
            lambda t: t["variables"].update(C=[1, 1]),
            lambda t: t["variables"].update(C=[1, False]),
            lambda t: t["variables"].update(G=[1, 0, 2]),
        ):
            trial = deepcopy(schema)
            change(trial)
            with self.subTest(domain=trial["variables"]):
                # Do not infer an inconsistent or non-binary domain.
                reasons = []
                answer = parse_extracted_constraints(
                    question, packet(schema=trial, evidence=evidence),
                    rejection_reason_sink=reasons)
                self.assertIsNone(answer)
                self.assertNotEqual(reasons, ["accepted"])

        # A unique case-only alias now deliberately canonicalizes to A.
        trial = deepcopy(schema)
        trial["bool_rules"][0]["args"][0]["args"][0] = "a"
        reasons = []
        expected = parse_extracted_constraints(
            question, packet(schema=schema, evidence=evidence))
        self.assertEqual(parse_extracted_constraints(
            question, packet(schema=trial, evidence=evidence),
            rejection_reason_sink=reasons), expected)
        self.assertEqual(reasons, ["accepted"])

    def test_no_implicit_boolean_domain_without_explicit_source_declaration(self):
        question = ("How many assignments to A and B are possible? "
                    "A must be less than B.")
        schema = {"variables": {"A": [0, 1], "B": [0, 1]},
                  "rules": [], "bool_rules": [
                      {"op": "and", "args": ["A", "B"]}], "query": "count"}
        evidence = ["How many assignments to A and B are possible?",
                    "A must be less than B."]
        reason = []
        self.assertIsNone(parse_extracted_constraints(
            question, packet(schema=schema, evidence=evidence),
            rejection_reason_sink=reason))
        self.assertEqual(reason, ["solver_type_error"])

    def test_model_review_must_demand_unambiguous_final_envelope(self):
        from neral_tir_experimental import run_experimental_tir
        original = messages("What does this program print? (Python)\nprint(2+3)")
        decisions = []
        replies = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(2+3)"}',
            "NERAL_FINAL_TEXT: 5",
        ])
        def model(turns):
            decisions.append(turns)
            return next(replies)
        result = run_experimental_tir(
            original, model, tool_runner=lambda *_: {
                "status": "PASSED", "output": "5", "verification_scope": "execution_only"})
        self.assertIn("NERAL_FINAL_TEXT: FINAL_ANSWER:",
                      decisions[1][-1]["content"])
        self.assertTrue(result["tool_output_reviewed"])

    def test_exact_source_python_stdout_is_adopted_if_llm_review_abstains(self):
        question = (
            "What exact integer does this Python program print? "
            "You may execute the program in an offline sandbox:\n"
            "data = [1, 3, 5]\n"
            "for i in range(len(data)):\n"
            "    data[i] += i\n"
            "print(sum(data))"
        )
        code = question.split("sandbox:\n", 1)[1]
        replies = iter([
            "NERAL_EXECUTE_JSON: " + json.dumps({
                "tool": "python_sandbox", "code": code}),
            "NERAL_FINAL_TEXT: UNVERIFIED",
        ])
        events = []
        def model(turns):
            events.append(turns)
            return next(replies)
        result = run_experimental_tir(
            messages(question + SUFFIX), model,
            extract_constraints=lambda _: '{"kind":"none"}',
            tool_runner=lambda *_: {
                "status": "PASSED", "output": "12",
                "verification_scope": "execution_only",
            })
        # Running the EXACT user-supplied deterministic program proves its print.
        self.assertEqual(result["text"], "FINAL_ANSWER: 12")
        self.assertEqual(result["verification_status"], "verified")
        self.assertTrue(result["verified"])
        self.assertTrue(result["tool_output_reviewed"])
        self.assertEqual(result["verification_scope"], "exact_user_python_execution")
        self.assertEqual(result["tool_used"], "neral_python_sandbox")
        self.assertEqual(result["tool_calls_succeeded"], 1)
        self.assertEqual(len(events), 0)  # Unchanged source bypasses Qwen entirely

    def test_modified_code_does_not_get_verified_from_stdout(self):
        question = ("What exact integer does this Python program print?\n"
                    "print(2+3)")
        replies = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(99)"}',
            "NERAL_FINAL_TEXT: UNVERIFIED",
            "FINAL_ANSWER: 5",
        ])
        result = run_experimental_tir(
            messages(question), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": "99"})
        self.assertFalse(result["verified"])
        self.assertFalse(result["tool_output_reviewed"])
        self.assertEqual(result["verification_status"], "base_fallback_unverified")

    def test_non_deterministic_source_code_is_not_automatically_verified(self):
        question = ("What does this Python program print?\n"
                    "import random\nprint(random.randint(1,9))")
        code = "import random\nprint(random.randint(1,9))"
        replies = iter([
            "NERAL_EXECUTE_JSON: " + json.dumps({
                "tool": "python_sandbox", "code": code}),
            "NERAL_FINAL_TEXT: UNVERIFIED",
            "I cannot verify the future random outcome.",
        ])
        result = run_experimental_tir(
            messages(question), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": "7"})
        self.assertFalse(result["verified"])
        self.assertFalse(result["tool_output_reviewed"])


    def test_forged_exact_source_provenance_is_rejected_by_packet_gate(self):
        from scripts.neral_same_base_live import checked_high_packet, PilotBlocked
        valid = {
            "mode": "HIGH", "model": "qwen3.5:9b",
            "inference_profile": "measured_nonthinking_2048",
            "model_origin": "local_foundation_not_neral_adapter",
            "adapter_loaded": False,
            "text": "FINAL_ANSWER: 12",
            "tool_used": "neral_python_sandbox",
            "model_generated": False,
            "tool_output_reviewed": True,
            "verification_scope": "exact_user_python_execution",
            "verification_status": "verified",
            "review_method": "exact_user_ast_match",
            "verified": True,
            "tool_calls_triggered": 1, "tool_calls_succeeded": 1,
        }
        self.assertEqual(checked_high_packet(valid), "FINAL_ANSWER: 12")
        for mutation in (
            {"tool_used": "neral_math_exact"},
            {"tool_output_reviewed": False},
            {"review_method": "model_review"},
            {"verification_status": "base_fallback_unverified"},
            {"verified": False},
            {"verification_scope": "exact_user_python_execution",
             "tool_calls_succeeded": 0},
            {"tool_calls_triggered": 0},
            {"model_generated": True},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(PilotBlocked):
                checked_high_packet({**valid, **mutation})

    def test_bool_source_declarations_are_case_sensitive_and_do_not_guess(self):
        q = ("How many Boolean assignments to Alpha,Beta satisfy ALL: "
             "Alpha AND Beta is True? Count assignments.")
        schema = {
            "variables": {"Alpha": [0, 1]},
            "rules": [],
            "bool_rules": [{"op": "and", "args": ["Alpha", "beta"]}],
            "query": "count",
        }
        pkt = packet(
            schema=schema,
            evidence=["How many Boolean assignments to Alpha,Beta",
                      "Alpha AND Beta is True", "Count assignments."])
        reason = []
        self.assertIsNone(parse_extracted_constraints(
            q, pkt, rejection_reason_sink=reason))
        self.assertNotEqual(reason, ["accepted"])

    def test_bool_inference_does_not_retype_numeric_constraint_variable(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is less than B; A AND B is True? Count assignments.")
        schema = {
            "variables": {"A": [0, 1], "B": [0, 1]},
            "rules": [{"left": "A", "op": "<", "right": "B"}],
            "bool_rules": [{"op": "and", "args": ["A", "B"]}],
            "query": "count",
        }
        reason = []
        self.assertIsNone(parse_extracted_constraints(
            q, packet(schema=schema, evidence=[
                "How many Boolean assignments to A,B",
                "A is less than B",
                "A AND B is True", "Count assignments."]),
            rejection_reason_sink=reason))
        self.assertEqual(reason, ["solver_type_error"])

    def test_valid_review_envelope_uses_observed_result_but_is_not_global_verification(self):
        responses = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(2+3)"}',
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 5",
        ])
        result = run_experimental_tir(
            messages("How many ways can X be arranged?" + SUFFIX),
            lambda _: next(responses),
            tool_runner=lambda *_: {"status": "PASSED", "output": "5"})
        self.assertEqual(result["text"], "FINAL_ANSWER: 5")
        self.assertTrue(result["tool_output_reviewed"])
        self.assertEqual(result["verification_status"], "tool_observed_unverified")
        self.assertFalse(result["verified"])


class AstLeafDiagnosticsAndReviewedProofCPU(unittest.TestCase):
    """Safe CPU-only replays; the private 374-token GPU proposal was not saved."""

    @staticmethod
    def _boolean_packet(question, variables, bool_rule, *, rules=None, evidence=None):
        return json.dumps({
            "kind": "formal", "complete": True,
            "schema": {
                "variables": variables, "rules": rules or [],
                "bool_rules": [bool_rule], "query": "count",
            },
            "evidence": evidence or [
                "How many Boolean assignments to A,B",
                "A XOR B is True",
                "Return one integer",
            ],
        }, separators=(",", ":"))

    def test_bool_ast_casefold_alias_is_canonicalized(self):
        question = ("How many Boolean assignments to A,B satisfy ALL: "
                    "A XOR B is True? Return one integer.")
        generated = self._boolean_packet(
            question, {"A": [False, True], "B": [False, True]},
            {"op": "xor", "args": ["A", "b"]},
        )
        reasons, details = [], []
        actual = parse_extracted_constraints(
            question, generated, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details)
        self.assertEqual(actual, 2)
        self.assertEqual(reasons, ["accepted"])
        self.assertEqual(details, [])
        self.assertNotIn('"b"', str(details))

    def test_bool_ast_declared_non_boolean_domain_gets_scalar_type_code(self):
        question = ("How many Boolean assignments to A,B satisfy ALL: "
                    "A XOR B is True? Return one integer.")
        generated = self._boolean_packet(
            question, {"A": [False, True], "B": [1, 2]},
            {"op": "xor", "args": ["A", "B"]},
        )
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            question, generated, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_type_error"])
        self.assertEqual(details, ["non_boolean_domain"])

    def test_bool_ast_missing_declared_source_variable_after_safe_inference_refusal(self):
        question = ("How many Boolean assignments to A,B,C satisfy ALL: "
                    "A XOR B is True; C is less than 2; "
                    "C AND A is True? Return one integer.")
        generated = self._boolean_packet(
            question, {"A": [False, True], "C": [1, 2]},
            {"op": "and", "args": ["A", "B", "C"]},
            rules=[{"left": "C", "op": "<", "value": 2}],
            evidence=[
                "How many Boolean assignments to A,B,C",
                "A XOR B is True",
                "C is less than 2",
                "C AND A is True",
                "Return one integer",
            ],
        )
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            question, generated, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_type_error"])
        self.assertEqual(details, ["missing_boolean_declaration"])

    def test_bool_ast_existing_source_declared_missing_variable_still_auto_repairs(self):
        question = ("How many Boolean assignments to A,B satisfy ALL: "
                    "A XOR B is True? Return one integer.")
        generated = self._boolean_packet(
            question, {"A": [False, True]},
            {"op": "xor", "args": ["A", "B"]},
        )
        reasons, details = [], []
        result = parse_extracted_constraints(
            question, generated, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details)
        self.assertEqual(result, 2)
        self.assertEqual(reasons, ["accepted"])
        self.assertEqual(details, [])

    def test_ast_scalar_diagnostic_propagates_without_variable_names(self):
        from scripts.neral_tir_hard_live import safe_first_turn_event
        question = ("How many Boolean assignments to A,B satisfy ALL: "
                    "A XOR B is True? Return one integer.")
        generated = self._boolean_packet(
            question, {"A": [False, True], "B": [False, True]},
            {"op": "xor", "args": ["A", "bb_secret"]},
        )
        rejection, details = [], []
        result = run_experimental_tir(
            messages(question), lambda _: "FINAL_ANSWER: 0",
            extract_constraints=lambda _: generated,
            extraction_rejection_sink=rejection,
            solver_rejection_detail_sink=details,
            first_turn_diagnostic=lambda: {
                "eval_count": 374, "prompt_eval_count": 839,
                "done_reason": "stop",
            },
        )
        event = safe_first_turn_event(result["first_turn_telemetry"])
        self.assertEqual(event["extraction_rejection_reason"], "solver_type_error")
        self.assertEqual(event["solver_rejection_detail"],
                         "completely_undeclared_name")
        self.assertNotIn("bb_secret", str(event))
        for injected in ("bb_secret", "unknown_boolean_reference:bb_secret"):
            with self.subTest(injected=injected):
                bad = {**event, "solver_rejection_detail": injected}
                self.assertIsNone(safe_first_turn_event(bad))

    def test_regular_model_review_is_verified_only_with_exact_original_python_proof(self):
        question = (
            "What exact integer does this Python program print? "
            "You may execute the program in an offline sandbox:\n"
            "data = [1, 3, 5]\n"
            "for i in range(len(data)):\n"
            "    data[i] += i\n"
            "print(sum(data))"
        )
        code = question.split("sandbox:\n", 1)[1]
        replies = iter([
            "NERAL_EXECUTE_JSON: " + json.dumps({
                "tool": "python_sandbox", "code": code}),
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 12",
        ])
        result = run_experimental_tir(
            messages(question + SUFFIX), lambda _: next(replies),
            extract_constraints=lambda _: '{"kind":"none"}',
            tool_runner=lambda *_: {
                "status": "PASSED", "output": "12",
                "verification_scope": "execution_only",
            },
        )
        self.assertEqual(result["tool_calls_succeeded"], 1)
        self.assertTrue(result["tool_output_reviewed"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["verification_status"], "verified")
        self.assertEqual(result["verification_scope"],
                         "exact_user_python_execution")
        self.assertEqual(result["review_method"],
                         "exact_user_ast_match")
        self.assertEqual(result["text"], "FINAL_ANSWER: 12")

    def test_model_review_disagreeing_with_tool_output_must_not_be_verified(self):
        question = ("What exact integer does this Python program print? "
                    "You may execute the program in an offline sandbox:\n"
                    "print(2+3)")
        code = "print(2+3)"
        replies = iter([
            "NERAL_EXECUTE_JSON: " + json.dumps({
                "tool": "python_sandbox", "code": code}),
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 99",
        ])
        result = run_experimental_tir(
            model_trace_messages(question + SUFFIX), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": "5"},
        )
        self.assertEqual(result["text"], "FINAL_ANSWER: 99")
        self.assertTrue(result["tool_output_reviewed"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["verification_status"],
                         "tool_observed_unverified")
        self.assertNotEqual(result.get("review_method"),
                            "exact_user_ast_match")

    def test_regular_review_of_generated_code_is_not_verified_by_exit_zero(self):
        question = ("What exact integer does this Python program print? "
                    "You may execute the program in an offline sandbox:\n"
                    "print(2+3)")
        replies = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(6)"}',
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 6",
        ])
        result = run_experimental_tir(
            model_trace_messages(question + SUFFIX), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": "6"},
        )
        self.assertTrue(result["tool_output_reviewed"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["verification_status"],
                         "tool_observed_unverified")


class AstCanonicalNamesAndSourceCodeReviewCPU(unittest.TestCase):
    """CPU regressions for unambiguous aliases and exact supplied-code AST."""

    @staticmethod
    def document(variables, tree):
        return json.dumps({
            "kind": "formal", "complete": True,
            "schema": {"variables": variables, "rules": [],
                       "bool_rules": [tree], "query": "count"},
            "evidence": ["How many Boolean assignments to Alpha,Beta",
                         "Alpha XOR Beta is True", "Return one integer"],
        })

    def test_casefold_and_boundary_whitespace_are_canonicalized(self):
        question = ("How many Boolean assignments to Alpha,Beta satisfy ALL: "
                    "Alpha XOR Beta is True? Return one integer")
        doc = self.document({"Alpha": [False, True], "Beta": [False, True]},
                            {"op": "xor", "args": [" alpha ", "\tBETA "]})
        reasons, details = [], []
        self.assertEqual(parse_extracted_constraints(
            question, doc, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details), 2)
        self.assertEqual(reasons, ["accepted"])
        self.assertEqual(details, [])

    def test_recursive_operator_alias_normalization(self):
        question = ("How many Boolean assignments to Alpha,Beta satisfy ALL: "
                    "Alpha XOR Beta is True? Return one integer")
        doc = self.document({"Alpha": [False, True], "Beta": [False, True]},
                            {"op": "not", "args": [
                                {"op": "implies", "args": [" ALPHA ", " beta "]}]})
        self.assertEqual(parse_extracted_constraints(question, doc), 1)

    def test_casefold_collision_is_rejected_and_reported_safely(self):
        question = ("How many Boolean assignments to Alpha,alpha satisfy ALL: "
                    "Alpha XOR alpha is True? Return one integer")
        doc = json.dumps({
            "kind": "formal", "complete": True,
            "schema": {"variables": {"Alpha": [False, True], "alpha": [False, True]},
                       "rules": [],
                       "bool_rules": [{"op": "xor", "args": [" ALPHA ", "alpha"]}],
                       "query": "count"},
            "evidence": ["How many Boolean assignments to Alpha,alpha",
                         "Alpha XOR alpha is True", "Return one integer"],
        })
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            question, doc, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_type_error"])
        self.assertEqual(details, ["casing_mismatch"])

    def test_no_canonical_match_reports_completely_undeclared_name(self):
        question = ("How many Boolean assignments to Alpha,Beta satisfy ALL: "
                    "Alpha XOR Beta is True? Return one integer")
        private = "model_secret_identifier"
        doc = self.document({"Alpha": [False, True], "Beta": [False, True]},
                            {"op": "xor", "args": ["Alpha", private]})
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            question, doc, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_type_error"])
        self.assertEqual(details, ["completely_undeclared_name"])
        self.assertNotIn(private, str(details))

    def test_non_boolean_declaration_never_gets_coerced(self):
        question = ("How many Boolean assignments to Alpha,Beta satisfy ALL: "
                    "Alpha XOR Beta is True? Return one integer")
        doc = self.document({"Alpha": [1, 2], "Beta": [False, True]},
                            {"op": "xor", "args": [" alpha ", "Beta"]})
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            question, doc, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(details, ["non_boolean_domain"])

    def test_exact_source_ast_match_ignores_markdown_comments_crlf_and_indent(self):
        fence = chr(96)*3
        question = (
            "What exact integer does this Python program print?\n"
            + fence + "python\n"
            + "    values=[2,3]\n"
            + "    # unchanged computation\n"
            + "    print(sum(values))\n"
            + fence
        )
        code = "values = [2,3]\r\nprint( sum(values) )\r\n# same tree"
        replies = iter([
            "NERAL_EXECUTE_JSON: " + json.dumps({
                "tool": "python_sandbox", "code": code}),
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 5",
        ])
        result = run_experimental_tir(
            messages(question + SUFFIX), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": "5"})
        self.assertTrue(result["verified"])
        self.assertEqual(result["verification_status"], "verified")
        self.assertEqual(result["review_method"], "exact_user_ast_match")

    def test_plain_code_ast_match_ignores_trailing_comments(self):
        question = ("What exact integer does this Python program print? "
                    "You may execute it:\nx=2\nprint(x+3)")
        code = "# harmless comment\nx = 2\nprint( x + 3 )"
        replies = iter([
            "NERAL_EXECUTE_JSON: " + json.dumps({
                "tool": "python_sandbox", "code": code}),
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 5",
        ])
        result = run_experimental_tir(
            messages(question + SUFFIX), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": "5"})
        self.assertTrue(result["verified"])
        self.assertEqual(result["review_method"], "exact_user_ast_match")

    def test_changed_code_is_not_verified_and_diagnoses_ast_mismatch(self):
        question = ("What exact integer does this Python program print? "
                    "You may execute it:\nx=2\nprint(x+3)")
        replies = iter([
            "NERAL_EXECUTE_JSON: " + json.dumps({
                "tool": "python_sandbox", "code": "x=2\nprint(x+4)"}),
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 6",
        ])
        result = run_experimental_tir(
            model_trace_messages(question + SUFFIX), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": "6"})
        self.assertFalse(result["verified"])
        self.assertEqual(result["verification_status"], "tool_observed_unverified")
        self.assertEqual(result.get("review_match_reason"), "body_statement_modified")

    def test_untrusted_random_program_cannot_be_verified_even_if_output_matches(self):
        question = ("What does this Python program print?\n"
                    "import random\nprint(random.randint(1,9))")
        code = "import random\nprint(random.randint(1,9))"
        replies = iter([
            "NERAL_EXECUTE_JSON: " + json.dumps({
                "tool": "python_sandbox", "code": code}),
            "NERAL_FINAL_TEXT: 5",
        ])
        result = run_experimental_tir(
            messages(question), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": "5"})
        self.assertFalse(result["verified"])

    def test_wrong_review_number_never_verified_despite_identical_ast(self):
        question = ("What exact integer does this Python program print? "
                    "You may execute it:\nprint(2+3)")
        replies = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(2+3)"}',
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 6",
        ])
        result = run_experimental_tir(
            model_trace_messages(question + SUFFIX), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": "5"})
        self.assertFalse(result["verified"])
        self.assertNotEqual(result.get("review_method"), "exact_user_ast_match")


class CpuAutoBooleanDeclarationAndReviewReason(unittest.TestCase):
    """CPU-only regression on source-grounded Boolean declarations and safe review reasons."""

    @staticmethod
    def seven_case():
        question = (
            "Let A,B,C,D,E,F,G be Boolean variables. "
            "Count the assignments satisfying (A XOR C XOR F) is True; "
            "(B OR D) is True; E = NOT F; (A AND G) is False; "
            "C implies G. Return one integer."
        )
        schema = {
            "variables": [{"name": name, "domain": [False, True]}
                          for name in "ABCDEF"],
            "rules": [],
            "bool_rules": [
                {"op": "xor", "args": ["A", "C", "F"]},
                {"op": "or", "args": ["B", "D"]},
                {"op": "eq", "args": ["E", {"op": "not", "args": ["F"]}]},
                {"op": "not", "args": [{"op": "and", "args": ["A", "G"]}]},
                {"op": "implies", "args": ["C", "G"]},
            ],
            "query": "count",
        }
        evidence = [
            "Let A,B,C,D,E,F,G be Boolean variables",
            "(A XOR C XOR F) is True",
            "(B OR D) is True",
            "E = NOT F",
            "(A AND G) is False",
            "C implies G",
            "Return one integer",
        ]
        return question, schema, evidence

    @staticmethod
    def packet(schema, evidence):
        return json.dumps(
            {"kind": "formal", "complete": True,
             "schema": schema, "evidence": evidence},
            separators=(",", ":"),
        )

    def test_missing_seventh_boolean_declared_in_source_is_registered(self):
        q, schema, evidence = self.seven_case()
        reasons, details = [], []
        answer = parse_extracted_constraints(
            q, self.packet(schema, evidence),
            rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details)
        from copy import deepcopy
        complete = deepcopy(schema)
        complete["variables"].append({"name": "G", "domain": [False, True]})
        baseline = parse_extracted_constraints(
            q, self.packet(complete, evidence))
        self.assertIsInstance(baseline, int)
        self.assertEqual(answer, baseline)
        self.assertEqual(reasons, ["accepted"])
        self.assertEqual(details, [])

    def test_unknown_eighth_identifier_not_in_source_is_not_auto_declared(self):
        q, schema, evidence = self.seven_case()
        schema["bool_rules"][0]["args"][-1] = "H"
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            q, self.packet(schema, evidence),
            rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_type_error"])
        self.assertEqual(details, ["completely_undeclared_name"])

    def test_reserved_operator_identifier_is_not_auto_declared(self):
        q, schema, evidence = self.seven_case()
        # Even when the source mentions an operator, it is not a variable.
        q += " AND is a logical connective."
        schema["bool_rules"][0]["args"][-1] = "AND"
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            q, self.packet(schema, evidence),
            rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertEqual(reasons, ["solver_type_error"])
        self.assertEqual(details, ["completely_undeclared_name"])

    def test_numeric_relation_context_prevents_boolean_auto_declare(self):
        q, schema, evidence = self.seven_case()
        q += " G must be less than F."
        evidence.append("G must be less than F")
        schema["rules"].append({"left": "G", "op": "<", "right": "F"})
        reasons, details = [], []
        self.assertIsNone(parse_extracted_constraints(
            q, self.packet(schema, evidence),
            rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details))
        self.assertNotEqual(reasons, ["accepted"])

    def test_eight_variable_hard_limit_not_bypassed(self):
        q, schema, evidence = self.seven_case()
        q = q.replace("A,B,C,D,E,F,G", "A,B,C,D,E,F,G,H,I")
        q = q.replace("C implies G", "C implies G; H XOR I is True")
        evidence = [e.replace("A,B,C,D,E,F,G", "A,B,C,D,E,F,G,H,I")
                    .replace("C implies G", "C implies G; H XOR I is True")
                    for e in evidence]
        schema["bool_rules"].append({"op": "xor", "args": ["H", "I"]})
        reasons = []
        self.assertIsNone(parse_extracted_constraints(
            q, self.packet(schema, evidence),
            rejection_reason_sink=reasons))
        self.assertNotEqual(reasons, ["accepted"])

    @staticmethod
    def review(question, code, result, suffix=True):
        responses = iter([
            "NERAL_EXECUTE_JSON: " + json.dumps({
                "tool": "python_sandbox", "code": code}),
            "NERAL_FINAL_TEXT: FINAL_ANSWER: " + result
            if suffix else "NERAL_FINAL_TEXT: " + result,
        ])
        return run_experimental_tir(
            model_trace_messages(question + (SUFFIX if suffix else "")),
            lambda _: next(responses),
            tool_runner=lambda *_: {"status": "PASSED", "output": result},
        )

    def test_source_absent_reason_visible_on_successful_review(self):
        out = self.review(
            "What exact integer does this Python program print? print(2+3)",
            "print(2+3)", "5",
        )
        self.assertTrue(out["tool_output_reviewed"])
        self.assertFalse(out["verified"])
        self.assertEqual(out["review_match_reason"], "user_source_not_extracted")

    def test_user_code_parse_reason_is_distinct(self):
        out = self.review(
            "What exact integer does this Python program print:\n"
            "for x in :\n    print(x)",
            "print(5)", "5",
        )
        self.assertFalse(out["verified"])
        self.assertEqual(out["review_match_reason"], "user_code_parse_error")

    def test_executed_code_parse_reason_is_distinct(self):
        out = self.review(
            "What exact integer does this Python program print:\nprint(5)",
            "print(", "5",
        )
        self.assertFalse(out["verified"])
        self.assertEqual(out["review_match_reason"], "executed_code_parse_error")

    def test_ast_structure_mismatch_reason_is_distinct(self):
        out = self.review(
            "What exact integer does this Python program print:\nprint(5)",
            "print(6)", "6",
        )
        self.assertFalse(out["verified"])
        self.assertEqual(out["review_match_reason"], "body_statement_modified")

    def test_matching_ast_keeps_verified_and_exact_user_ast_match(self):
        out = self.review(
            "What exact integer does this Python program print:\n"
            "values = [1,2]\nprint(sum(values))",
            "# comment\nvalues=[1, 2]\nprint( sum(values) )", "3",
        )
        self.assertTrue(out["verified"])
        self.assertEqual(out["verification_status"], "verified")
        self.assertEqual(out["review_method"], "exact_user_ast_match")
        self.assertTrue(out["tool_output_reviewed"])

    def test_fallback_review_reason_does_not_leak_input_text(self):
        q = "What exact integer does this Python program print: secret_417?"
        out = self.review(q, "print(5)", "5")
        self.assertNotIn("secret_417", str(out.get("review_match_reason")))


class BooleanTokensAndPrintOnlyReviewCPU(unittest.TestCase):
    """Regression cases, not a reconstruction of model-generated GPU schemas."""

    @staticmethod
    def run_boolean(question, variables, rules, evidence):
        proposal = json.dumps({
            "kind": "formal", "complete": True,
            "schema": {"variables": variables, "rules": [],
                       "bool_rules": rules, "query": "count"},
            "evidence": evidence,
        }, separators=(",", ":"))
        reasons, details, tags = [], [], []
        answer = parse_extracted_constraints(
            question, proposal, rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details,
            ast_check_code_sink=tags)
        return answer, reasons, details, tags

    def test_boolean_literal_strings_and_numeric_tokens_are_constants_not_names(self):
        for literal, truth in (
            ("true", True), ("True", True), ("false", False),
            ("False", False), ("1", True), ("0", False),
            (1, True), (0, False),
        ):
            with self.subTest(token=literal):
                state = "True" if truth else "False"
                question = ("How many Boolean assignments to A,B satisfy ALL: "
                            "A XOR B is True; A is " + state +
                            "? Return one integer.")
                answer, reasons, details, tags = self.run_boolean(
                    question, {"A": [False, True], "B": [False, True]},
                    [{"op": "xor", "args": ["A", "B"]},
                     {"op": "eq", "args": ["A", literal]}],
                    ["How many Boolean assignments to A,B",
                     "A XOR B is True", "A is " + state,
                     "Return one integer"])
                self.assertEqual(answer, 1)
                self.assertEqual(reasons, ["accepted"])
                self.assertEqual(details, [])
                self.assertEqual(tags, ["boolean_literal_token"])

    def test_declared_variable_named_true_is_not_silently_retyped(self):
        question = ("How many Boolean assignments to true,B satisfy ALL: "
                    "true XOR B is True? Return one integer.")
        result = self.run_boolean(
            question, {"true": [False, True], "B": [False, True]},
            [{"op": "xor", "args": ["true", "B"]}],
            ["How many Boolean assignments to true,B",
             "true XOR B is True", "Return one integer"])
        self.assertEqual(result, (2, ["accepted"], [], []))

    def test_initials_map_only_with_unique_explicit_boolean_names(self):
        question = ("How many Boolean assignments to Alice,Bob satisfy ALL: "
                    "Alice XOR Bob is True? Return one integer.")
        result = self.run_boolean(
            question, {"Alice": [False, True], "Bob": [False, True]},
            [{"op": "xor", "args": ["A", "B"]}],
            ["How many Boolean assignments to Alice,Bob",
             "Alice XOR Bob is True", "Return one integer"])
        self.assertEqual(result, (2, ["accepted"], [], ["initial_letter_token"]))

    def test_duplicate_initials_remain_ambiguous(self):
        question = ("How many Boolean assignments to Alice,Anna satisfy ALL: "
                    "Alice XOR Anna is True? Return one integer.")
        result = self.run_boolean(
            question, {"Alice": [False, True], "Anna": [False, True]},
            [{"op": "xor", "args": ["A", "Anna"]}],
            ["How many Boolean assignments to Alice,Anna",
             "Alice XOR Anna is True", "Return one integer"])
        self.assertEqual(result[0], None)
        self.assertEqual(result[1], ["solver_type_error"])
        self.assertEqual(result[2], ["completely_undeclared_name"])
        self.assertEqual(result[3], ["initial_letter_token"])

    def test_unknown_name_does_not_auto_register_and_reports_scalar(self):
        question = ("How many Boolean assignments to Alice,Bob satisfy ALL: "
                    "Alice XOR Bob is True? Return one integer.")
        result = self.run_boolean(
            question, {"Alice": [False, True], "Bob": [False, True]},
            [{"op": "xor", "args": ["Alice", "CONFIDENTIAL_Z99"]}],
            ["How many Boolean assignments to Alice,Bob",
             "Alice XOR Bob is True", "Return one integer"])
        self.assertEqual(result[0], None)
        self.assertEqual(result[1], ["solver_type_error"])
        self.assertEqual(result[2], ["completely_undeclared_name"])
        self.assertEqual(result[3], ["unmapped_identifier"])
        self.assertNotIn("CONFIDENTIAL_Z99", str(result[1:]))

    def test_no_initial_mapping_without_explicit_source_boolean_introduction(self):
        question = ("How many assignments to Alice,Bob satisfy ALL: "
                    "Alice XOR Bob is True? Return one integer.")
        result = self.run_boolean(
            question, {"Alice": [False, True], "Bob": [False, True]},
            [{"op": "xor", "args": ["A", "B"]}],
            ["How many assignments to Alice,Bob",
             "Alice XOR Bob is True", "Return one integer"])
        self.assertIsNone(result[0])
        self.assertEqual(result[3], ["unmapped_identifier"])

    @staticmethod
    def review(question, code, answer):
        replies = iter([
            "NERAL_EXECUTE_JSON: " +
            json.dumps({"tool": "python_sandbox", "code": code}),
            "NERAL_FINAL_TEXT: FINAL_ANSWER: " + answer,
        ])
        return run_experimental_tir(
            model_trace_messages(question + SUFFIX), lambda _: next(replies),
            tool_runner=lambda *_: {"status": "PASSED", "output": answer})

    def test_only_appended_print_of_final_assigned_name_is_verifiable(self):
        question = ("What integer does this program output after execution:\n"
                    "value = 2 + 3")
        result = self.review(question, "value=2+3\nprint(value)", "5")
        self.assertTrue(result["verified"])
        self.assertTrue(result["tool_output_reviewed"])
        self.assertEqual(result["review_method"], "exact_user_ast_match")
        self.assertEqual(result["verification_status"], "verified")

    def test_only_appended_print_of_last_pure_expression_is_verifiable(self):
        question = ("What is the value of the final Python expression? "
                    "What does it output:\nx=2\nx+3")
        result = self.review(question, "x = 2\nx + 3\nprint(x + 3)", "5")
        self.assertTrue(result["verified"])
        self.assertEqual(result["review_method"], "exact_user_ast_match")

    def test_appended_print_of_unrelated_constant_cannot_verify(self):
        question = ("What does this program output after execution:\nvalue = 2+3")
        result = self.review(question, "value = 2+3\nprint(999)", "999")
        self.assertFalse(result["verified"])
        self.assertEqual(result["review_match_reason"], "extra_trailing_statements")

    def test_changed_original_computation_plus_correct_print_is_not_verified(self):
        question = ("What does this program output after execution:\nvalue=2+3")
        result = self.review(question, "value=2+4\nprint(value)", "6")
        self.assertFalse(result["verified"])
        self.assertEqual(result["review_match_reason"], "body_statement_modified")

    def test_existing_print_cannot_gain_duplicate_print(self):
        question = ("What does this program print:\nprint(2+3)")
        result = self.review(question, "print(2+3)\nprint(5)", "5")
        self.assertFalse(result["verified"])
        self.assertEqual(result["review_match_reason"], "extra_trailing_statements")


    def test_boolean_ast_literal_success_code_survives_safe_first_turn_telemetry(self):
        from scripts.neral_tir_hard_live import safe_first_turn_event
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A XOR B is True; A is True? Return one integer.")
        proposed = json.dumps({
            "kind": "formal", "complete": True,
            "schema": {"variables": {"A": [False, True], "B": [False, True]},
                       "rules": [], "bool_rules": [
                           {"op": "xor", "args": ["A", "B"]},
                           {"op": "eq", "args": ["A", "true"]}],
                       "query": "count"},
            "evidence": ["How many Boolean assignments to A,B",
                         "A XOR B is True", "A is True", "Return one integer"],
        })
        result = run_experimental_tir(
            messages(q), lambda _: "FINAL_ANSWER: 0",
            extract_constraints=lambda _: proposed)
        self.assertEqual(result["extraction_status"], "accepted")
        event = safe_first_turn_event(result["first_turn_telemetry"])
        self.assertEqual(event["ast_check_code"], "boolean_literal_token")
        self.assertEqual(event["extraction_rejection_reason"], "accepted")
        self.assertNotIn("true", str(event.get("ast_check_code")))
        self.assertIsNone(safe_first_turn_event({
            **event, "ast_check_code": "boolean_literal_token:private_data"
        }))


    def test_terminal_pure_expression_may_be_wrapped_in_print(self):
        question = ("What integer does this Python program output:\n"
                    "x=2\nx+3")
        result = self.review(question, "x = 2\nprint(x+3)", "5")
        self.assertTrue(result["verified"])
        self.assertEqual(result["review_method"], "exact_user_ast_match")

    def test_initial_alias_success_reaches_safe_first_turn_telemetry(self):
        from scripts.neral_tir_hard_live import safe_first_turn_event
        q = ("How many Boolean assignments to Alice,Bob satisfy ALL: "
             "Alice XOR Bob is True? Return one integer.")
        proposed = json.dumps({
            "kind": "formal", "complete": True,
            "schema": {"variables": {"Alice": [False, True],
                                     "Bob": [False, True]},
                       "rules": [],
                       "bool_rules": [{"op": "xor", "args": ["A", "B"]}],
                       "query": "count"},
            "evidence": ["How many Boolean assignments to Alice,Bob",
                         "Alice XOR Bob is True", "Return one integer"],
        })
        result = run_experimental_tir(
            messages(q), lambda _: "FINAL_ANSWER: 0",
            extract_constraints=lambda _: proposed)
        self.assertEqual(result["extraction_status"], "accepted")
        event = safe_first_turn_event(result["first_turn_telemetry"])
        self.assertEqual(event["ast_check_code"], "initial_letter_token")

    def test_unmapped_identifier_failure_reaches_safe_first_turn_telemetry(self):
        from scripts.neral_tir_hard_live import safe_first_turn_event
        q = ("How many Boolean assignments to Alice,Bob satisfy ALL: "
             "Alice XOR Bob is True? Return one integer.")
        proposed = json.dumps({
            "kind": "formal", "complete": True,
            "schema": {"variables": {"Alice": [False, True],
                                     "Bob": [False, True]},
                       "rules": [],
                       "bool_rules": [{"op": "xor",
                                      "args": ["Alice", "secret_unknown_42"]}],
                       "query": "count"},
            "evidence": ["How many Boolean assignments to Alice,Bob",
                         "Alice XOR Bob is True", "Return one integer"],
        })
        result = run_experimental_tir(
            messages(q), lambda _: "FINAL_ANSWER: 0",
            extract_constraints=lambda _: proposed)
        self.assertEqual(result["extraction_status"], "fallback")
        event = safe_first_turn_event(result["first_turn_telemetry"])
        self.assertEqual(event["ast_check_code"], "unmapped_identifier")
        self.assertEqual(event["solver_rejection_detail"],
                         "completely_undeclared_name")
        self.assertNotIn("secret_unknown_42", str(event))
