from __future__ import annotations
import unittest

from frontier350.chat350_sft_format import encode_messages
from frontier350.chat350_preference_format import encode_final_response
from frontier350.chat_prompt350 import prompt_ids,render_history

class Enc:
    def __init__(self,ids): self.ids=ids

class CharTokenizer:
    specials={"<|bos|>":0,"<|eos|>":1}
    def token_to_id(self,x): return self.specials.get(x)
    def encode(self,text): return Enc([100+ord(c) for c in str(text)])

class BoundaryMergeTokenizer(CharTokenizer):
    """Toy BPE: merge across ': ' only when both chars are encoded together."""
    def encode(self, text):
        ids = []
        pos = 0
        text = str(text)
        while pos < len(text):
            if text.startswith(": ", pos):
                ids.append(9999)
                pos += 2
            else:
                ids.append(100 + ord(text[pos]))
                pos += 1
        return Enc(ids)


class FormatParityTests(unittest.TestCase):
    def test_runtime_prefix_uses_sft_bpe_boundaries(self):
        tok = BoundaryMergeTokenizer()
        messages = [
            {"role":"user","content":"Hallo"},
            {"role":"assistant","content":"Hi"},
            {"role":"user","content":"Wie heißt du?"},
            {"role":"assistant","content":"Neral."},
        ]
        sft = encode_messages(tok, messages, max_seq_len=1024)
        prefix = prompt_ids(tok, [(m["role"], m["content"]) for m in messages[:-1]])
        response = tok.encode(" Neral.\n").ids
        self.assertEqual(sft.input_ids, prefix + response)
        self.assertNotEqual(prefix, [0] + tok.encode(
            render_history([(m["role"], m["content"]) for m in messages[:-1]])
        ).ids)
        pref = encode_final_response(tok, messages[:-1], "Neral.", max_seq_len=1024)
        self.assertEqual(sft.input_ids, pref.input_ids)

    def test_runtime_token_count_matches_sft_multi_turn(self):
        tok = BoundaryMergeTokenizer()
        history=[("user","eins"),("assistant","zwei"),("user","drei")]
        self.assertEqual(prompt_ids(tok,history)[0],0)
        self.assertEqual(prompt_ids(tok,history)[1:],
                         tok.encode("User: eins\nAssistant:").ids +
                         tok.encode(" zwei\n").ids +
                         tok.encode("User: drei\nAssistant:").ids)


    def test_multiturn_sft_dpo_runtime_prefix_parity(self):
        tok=CharTokenizer()
        context=[
            {"role":"user","content":"Hallo"},
            {"role":"assistant","content":"Hi"},
            {"role":"user","content":"Wie geht es dir?"},
        ]
        answer="Gut."
        full=context+[{"role":"assistant","content":answer}]
        sft=encode_messages(tok,full,max_seq_len=1024)
        pref=encode_final_response(tok,context,answer,max_seq_len=1024)
        history=[(m["role"],m["content"]) for m in context]
        runtime_prefix=prompt_ids(tok,history)
        answer_ids=list(tok.encode(" "+answer+"\n").ids)

        self.assertEqual(sft.input_ids,pref.input_ids)
        self.assertEqual(sft.input_ids,runtime_prefix+answer_ids)
        self.assertEqual(sft.targets[-1],1)
        self.assertEqual(pref.targets[-1],1)

    def test_runtime_text_contract(self):
        self.assertEqual(
            render_history([("user","A"),("assistant","B"),("user","C")]),
            "User: A\nAssistant: B\nUser: C\nAssistant:"
        )

    def test_preference_masks_prior_assistant_but_sft_supervises_it(self):
        tok=CharTokenizer()
        context=[
            {"role":"user","content":"A"},
            {"role":"assistant","content":"B"},
            {"role":"user","content":"C"},
        ]
        full=context+[{"role":"assistant","content":"D"}]
        sft=encode_messages(tok,full,max_seq_len=1024)
        pref=encode_final_response(tok,context,"D",max_seq_len=1024)
        self.assertGreater(sft.assistant_tokens,pref.response_tokens)
        self.assertEqual(sft.input_ids,pref.input_ids)

if __name__=="__main__":
    unittest.main()
