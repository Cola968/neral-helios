from __future__ import annotations
import unittest
from frontier350.chat350_text_hygiene import clean_human_text,all_texts_safe
from frontier350.build_oasst2_multiturn350 import filter_cross_split_message_leakage

def row(rid,texts):
    msgs=[]
    role="user"
    for text in texts:
        msgs.append({"role":role,"content":text})
        role="assistant" if role=="user" else "user"
    if msgs[-1]["role"]!="assistant":
        msgs.append({"role":"assistant","content":"abschließende antwort "+rid})
    return {"id":rid,"language":"de","domain":"chat","source":"unit","license":"CC0","messages":msgs}

class HumanDataHygieneTests(unittest.TestCase):
    def test_secret_and_contact_rejected(self):
        self.assertEqual(clean_human_text("mail me at x@example.com")[1],"contact_or_secret")
        self.assertEqual(clean_human_text("token sk-"+"A"*24)[1],"contact_or_secret")
        self.assertEqual(clean_human_text("AKIA"+"A"*16)[1],"contact_or_secret")

    def test_clean_text_passes(self):
        self.assertEqual(clean_human_text("Eine normale hilfreiche Antwort."),("Eine normale hilfreiche Antwort.",None))
        self.assertTrue(all_texts_safe(["Frage","Antwort"])[0])

    def test_cross_split_long_message_filtered(self):
        shared="dies ist eine ausreichend lange identische nachricht"
        base={"train":[row("base-tr",[shared])],"val":[row("base-va",["andere ausreichend lange nachricht"])],"test":[row("base-te",["noch eine ausreichend lange nachricht"])]}
        add={"train":[],"val":[row("cand",[shared])],"test":[]}
        kept,stats=filter_cross_split_message_leakage(base,add,min_normalized_chars=20)
        self.assertEqual(len(kept["val"]),0)
        self.assertEqual(stats["cross_split_message_filtered"],1)

    def test_short_generic_message_not_used_as_leakage_key(self):
        base={"train":[row("base-tr",["Ja"])],"val":[row("base-va",["Nein"])],"test":[row("base-te",["Okay"])]}
        add={"train":[],"val":[row("cand",["Ja"])],"test":[]}
        kept,stats=filter_cross_split_message_leakage(base,add,min_normalized_chars=20)
        self.assertEqual(len(kept["val"]),1)
        self.assertEqual(stats["cross_split_message_filtered"],0)

if __name__=="__main__":
    unittest.main()
