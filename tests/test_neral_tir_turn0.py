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
