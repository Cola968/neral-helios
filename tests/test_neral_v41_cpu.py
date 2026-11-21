"""V4.1 migration contracts. CPU-only, no GPU, network, or user files."""
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from test_neral_three_tier_api import make_runtime
from neral_v41_context import resolve_question
from neral_verified_notes import SessionConversation, VerifiedStrategyStore
from neral_cpu_foundation.memory import LocalMemory


class V41TierTests(unittest.TestCase):
    def test_public_modes_and_amber_full_answer_budget(self):
        runtime, amber, nine_b, _ = make_runtime()
        turns = [{"role": "system", "content": "You are a chat assistant."},
                 {"role": "user", "content": "Erkläre mir die Stadtklimatologie."}]
        a = runtime.answer(turns, "Amber")
        q = runtime.answer(turns, "Quartz")
        o = runtime.answer(turns, "Obsidian")
        self.assertEqual([a["tier"], q["tier"], o["tier"]],
                         ["Amber", "Quartz", "Obsidian"])
        self.assertEqual(amber.calls[0]["max_tokens"], 768)
        self.assertEqual([c["max_tokens"] for c in nine_b.calls], [256, 768])
        self.assertFalse(a["verified"])

    def test_legacy_aliases_keep_original_behavior(self):
        runtime, low, high, _ = make_runtime()
        turns = [{"role": "system", "content": "test"},
                 {"role": "user", "content": "Hallo"}]
        self.assertEqual(runtime.answer(turns, "Luna")["tier"], "Luna")
        self.assertEqual(runtime.answer(turns, "Terra")["tier"], "Terra")
        self.assertEqual(runtime.answer(turns, "Helios")["tier"], "Helios")
        self.assertEqual(low.calls[0]["max_tokens"], 96)

    def test_context_followup_uses_user_provenance_only(self):
        turns = [{"role": "system", "content": "safe"},
                 {"role": "user", "content": "Wie lange fliegt man von Berlin nach Abu Dhabi?"},
                 {"role": "assistant", "content": "Es dauert angeblich exakt eine Stunde."},
                 {"role": "user", "content": "Und mit einem A321XLR?"},
                 {"role": "assistant", "content": "Das ist ein anderes Flugzeug."},
                 {"role": "user", "content": "Wie viel Treibstoff braucht er?"}]
        result = resolve_question(turns)
        self.assertTrue(result.is_followup)
        self.assertIn("Berlin nach Abu Dhabi", result.work_question)
        self.assertIn("A321XLR", result.work_question)
        self.assertIn("Treibstoff", result.work_question)
        self.assertNotIn("exakt eine Stunde", result.work_question)
        self.assertFalse(result.independent_proof_allowed)

    def test_context_topic_change_does_not_inherit_constraints(self):
        turns = [{"role":"system","content":"safe"},
                 {"role":"user","content":"Plane Schichten: Anna arbeitet 7 Stunden."},
                 {"role":"assistant","content":"Okay."},
                 {"role":"user","content":"Wie viele Teiler hat die Zahl 120?"}]
        result = resolve_question(turns)
        self.assertFalse(result.is_followup)
        self.assertEqual(result.work_question, "Wie viele Teiler hat die Zahl 120?")
        self.assertNotIn("Anna", result.work_question)
        self.assertTrue(result.independent_proof_allowed)

    def test_transient_session_memory_never_writes_disk(self):
        session = SessionConversation(max_turns=4)
        session.add("user", "Hallo")
        session.add("assistant", "Hallo")
        self.assertEqual(len(session.history()), 2)
        session.clear()
        self.assertEqual(session.history(), [])

    def test_personal_memory_requires_consent_and_can_be_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            memory=LocalMemory(Path(directory)/"personal.db")
            with self.assertRaises(PermissionError):
                memory.upsert("user","projekt","NERAL",consent=False)
            self.assertTrue(memory.upsert("user","projekt","NERAL",consent=True))
            self.assertEqual(len(memory.list("user")), 1)
            self.assertTrue(memory.forget("user","projekt"))
            self.assertFalse(memory.list("user"))
            memory.close()

    def test_verified_strategies_never_store_live_web_or_unverified_corrections(self):
        with tempfile.TemporaryDirectory() as directory:
            store=VerifiedStrategyStore(Path(directory)/"strategies.jsonl")
            proof={"agent_checked": True, "tool_verified": True, "verified": True,
                   "error_code": None, "text": "42",
                   "proof_backend": "trusted_source_arithmetic",
                   "tool_used": "python_sandbox",
                   "verified_question_sha256": sha256(b"Berechne 6*7").hexdigest()}
            self.assertFalse(store.learn("Wie ist der aktuelle Aktienkurs?",proof))
            self.assertFalse(store.learn("Berechne 6*7", {**proof,"verified":False}))
            self.assertTrue(store.learn("Berechne 6*7",proof))
            hint=store.hint("Wie berechnet man 7*9?")
            self.assertIn("Rechen",hint)
            self.assertFalse(store.accept_correction("Berechne 6*7","44",proof))
            self.assertFalse(store.accept_correction("Berechne 6*7","42",{
                **proof,"verified":False}))
            self.assertTrue(store.accept_correction("Berechne 6*7","42",proof))
            self.assertFalse(store.accept_correction("Berechne 7*7","42",proof))


if __name__ == "__main__":
    unittest.main()
