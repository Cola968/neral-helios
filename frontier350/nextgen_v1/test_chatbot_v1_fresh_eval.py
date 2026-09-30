from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))

import chatbot_v1_data as behavior
import phoenix_chatbot_v1_fresh_eval as fresh


class FreshEvalTests(unittest.TestCase):
    def test_suite_identity_and_unique_cases(self):
        self.assertEqual(len(fresh.CASES),24)
        ids=[row[0] for row in fresh.CASES]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(fresh.suite_sha()),64)

    def test_fresh_prompts_are_not_behavior_user_prompts(self):
        behavior_rows=behavior.build_split("train")+behavior.build_split("validation")
        behavior_prompts={behavior.norm(turn["user"]) for row in behavior_rows for turn in row["turns"]}
        for _ident,_kind,prompt,_spec in fresh.CASES:
            self.assertNotIn(behavior.norm(prompt),behavior_prompts)

    def test_objective_scorers(self):
        self.assertTrue(fresh.score("exact","  NEBEL  ","NEBEL"))
        self.assertFalse(fresh.score("exact","NEBEL.","NEBEL"))
        self.assertTrue(fresh.score("contains","Der Code ist RIVER-204.",("river-204",)))
        self.assertTrue(fresh.score("contains_not","Jetzt gilt BETA-92.",(("beta-92",),("alfa-31",))))
        self.assertFalse(fresh.score("contains_not","ALFA-31 wurde zu BETA-92.",(("beta-92",),("alfa-31",))))
        self.assertTrue(fresh.score("uncertain","Das kann ich nicht bestimmen.",("weiß","nicht","bestimm")))
        self.assertTrue(fresh.score("two_bullets","- Eins\n- Zwei",None))
        self.assertFalse(fresh.score("two_bullets","- Eins\n- Zwei\n- Drei",None))

    def test_suite_covers_core_chatbot_dimensions(self):
        kinds={row[1] for row in fresh.CASES}
        self.assertEqual(kinds,{"exact","contains","contains_not","uncertain","two_bullets"})
        ids={row[0].split("-")[0] for row in fresh.CASES}
        for expected in ("exact","math","state","correct","unknown","fact","translate","format","identity"):
            self.assertIn(expected,ids)


if __name__=="__main__":
    unittest.main()
