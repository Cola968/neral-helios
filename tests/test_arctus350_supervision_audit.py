"""CPU-only supervision-density audit; no weights, GPU, or real private corpus."""
import json
import unittest

from frontier350.chat350_sft_format import EncodedChatExample, encode_messages
from frontier350.arctus350_supervision_audit import audit_supervision, validate_encoded


class CharTokenizer:
    """Small deterministic synthetic tokenizer: never a native Arctus checkpoint."""
    def token_to_id(self,token):
        return {"<|bos|>":0,"<|eos|>":1,"<|sep|>":2,"<|pad|>":3}.get(token)

    class Encoding:
        def __init__(self,ids):self.ids=ids

    def encode(self,text):
        return self.Encoding([4+ord(ch) for ch in text])


def row(lang,domain,prompt,answer):
    return {"language":lang,"domain":domain,"messages":[
        {"role":"user","content":prompt},{"role":"assistant","content":answer}]}


class SupervisionAuditTests(unittest.TestCase):
    def setUp(self):self.tok=CharTokenizer()

    def encode(self,record,max_seq_len):
        return encode_messages(self.tok,record["messages"],max_seq_len=max_seq_len)

    def dataset(self):
        return {
            "train":[row("de","dialogue","Hallo","Guten Tag"),row("en","math","2+3","5")],
            "val":[row("de","math","Wie viel ist sieben plus acht?","15")],
            "test":[row("en","dialogue","Can you help?","Certainly.")],
        }

    def test_native_chat_format_counts_only_assistant_labels_and_final_eos(self):
        encoded=self.encode(row("de","dialogue","Hallo","Guten Tag"),512)
        counts=validate_encoded(encoded,eos_id=1,pad_id=3)
        self.assertEqual(counts["assistant_targets"],encoded.assistant_tokens)
        self.assertEqual(counts["eos_targets"],1)
        self.assertEqual(counts["context_targets"],len(encoded.targets)-encoded.assistant_tokens)
        self.assertEqual(counts["sequence_length"],len(encoded.input_ids))
        self.assertTrue(any(v==-100 for v in encoded.targets))

    def test_real_encoder_multiturn_aggregate_by_split_and_language(self):
        ds=self.dataset()
        ds["train"].append({"language":"de","domain":"dialogue",
            "messages":[{"role":"user","content":"Erste Frage"},
                        {"role":"assistant","content":"Erste Antwort"},
                        {"role":"user","content":"Zweite Frage"},
                        {"role":"assistant","content":"Zweite Antwort"}]})
        report=audit_supervision(ds,encode=self.encode,max_seq_len=512,eos_id=1,pad_id=3)
        self.assertEqual(report["schema"],"arctus350-supervision-audit-v1")
        self.assertEqual(report["splits"]["train"]["rows"],3)
        self.assertEqual(report["splits"]["train"]["eligible_rows"],3)
        self.assertEqual(report["splits"]["train"]["terminal_eos_targets"],3)
        self.assertIn("de",report["splits"]["train"]["languages"])
        self.assertIn("en",report["splits"]["train"]["languages"])
        self.assertGreater(report["splits"]["train"]["supervision_fraction"],0.0)
        self.assertLess(report["splits"]["train"]["supervision_fraction"],1.0)
        self.assertEqual(report["splits"]["test"]["rows"],1)
        self.assertEqual(report["status"],"AUDIT_ONLY")
        self.assertEqual(report["model_metrics"],"NOT_MEASURED")
        payload=json.dumps(report)
        for text in ("Erste Frage","Guten Tag","Zweite Antwort","Can you help?"):
            self.assertNotIn(text,payload)

    def test_oversized_examples_are_counted_not_silently_dropped(self):
        ds=self.dataset()
        report=audit_supervision(ds,encode=self.encode,max_seq_len=25,eos_id=1,pad_id=3)
        self.assertGreater(report["splits"]["val"]["ineligible_rows"],0)
        self.assertEqual(
            report["splits"]["val"]["eligible_rows"]+report["splits"]["val"]["ineligible_rows"],
            report["splits"]["val"]["rows"])

    def test_malformed_alignment_and_mask_leaks_fail_closed(self):
        bad=(
            EncodedChatExample([0,4],[4],1),
            EncodedChatExample([0,4],[1,-100],1),
            EncodedChatExample([0,4],[4,1],2),
            EncodedChatExample([0,4],[-100,-100],0),
            EncodedChatExample([0,3],[-100,1],1),
            EncodedChatExample([0,4],[-100,3],1),
            EncodedChatExample([0,4],[-100,1],99),
        )
        for example in bad:
            with self.subTest(example=example):
                with self.assertRaises(ValueError):
                    validate_encoded(example,eos_id=1,pad_id=3)

    def test_rejects_silent_generic_valueerror_and_invalid_metadata(self):
        ds=self.dataset()
        for broken in [
            lambda row,max_seq_len: (_ for _ in ()).throw(ValueError("bad tokenizer digest")),
            lambda row,max_seq_len: EncodedChatExample([0],[1],1),
        ]:
            with self.assertRaises(ValueError):
                audit_supervision(ds,encode=broken,max_seq_len=128,eos_id=1,pad_id=3)
        ds=self.dataset()
        del ds["val"]
        with self.assertRaises(ValueError):
            audit_supervision(ds,encode=self.encode,max_seq_len=512,eos_id=1,pad_id=3)
        ds=self.dataset();ds["train"][0]["language"]=""
        with self.assertRaises(ValueError):
            audit_supervision(ds,encode=self.encode,max_seq_len=512,eos_id=1,pad_id=3)

    def test_audit_is_deterministic_and_has_zero_gpu_or_model_dependency(self):
        a=audit_supervision(self.dataset(),encode=self.encode,max_seq_len=512,eos_id=1,pad_id=3)
        b=audit_supervision(self.dataset(),encode=self.encode,max_seq_len=512,eos_id=1,pad_id=3)
        self.assertEqual(a,b)
        self.assertEqual(a["compute"],"CPU, no model inference, no optimizer steps")
        self.assertNotIn("improvement_pct",json.dumps(a))


if __name__=="__main__":
    unittest.main()
