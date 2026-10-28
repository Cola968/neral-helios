"""CPU contracts for bounded, curriculum-balanced 2B native FP32 SFT."""
import unittest
from frontier350.arctus350_curriculum_fp32 import select_curriculum,admit_curriculum

class CurriculumContracts(unittest.TestCase):
    def fixture(self,each=5):
        return [
            (f"{domain}_{i}","de" if i%2==0 else "en",domain,
             "neral-generated-math-en" if domain=="math" else "neral-generated-de-core",
             "Project-generated deterministic data; no external corpus",i)
            for domain in ("math","code","format","chat","reasoning")
            for i in range(each)
        ]

    def test_no_data_reuse_or_external_corpus_and_domain_quota(self):
        rows=self.fixture(8)
        rows.append(("external","en","math","oasst2-human-en","Apache-2.0",4))
        quotas={"math":4,"code":4,"format":4,"chat":4,"reasoning":4}
        chosen=select_curriculum(rows,quotas)
        self.assertEqual(len(chosen),20)
        self.assertEqual(len({r[0] for r in chosen}),20)
        from collections import Counter
        self.assertEqual(dict(Counter(r[2] for r in chosen)),quotas)
        self.assertNotIn("external",[x[0] for x in chosen])
        self.assertEqual(chosen,select_curriculum(list(reversed(rows)),quotas))

    def test_insufficient_domain_fails_instead_of_silent_substitution(self):
        with self.assertRaisesRegex(ValueError,"insufficient"):
            select_curriculum(self.fixture(2),{"math":3})
        with self.assertRaises(ValueError):
            select_curriculum(self.fixture(2),{"math":0})
        with self.assertRaises(ValueError):
            select_curriculum(self.fixture(2),{"fiction":2})

    def test_invalid_or_duplicate_row_id_rejected(self):
        rows=self.fixture(5)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            select_curriculum(rows+[rows[0]],{"math":3})

    def test_gpu_and_absolute_training_ceiling(self):
        self.assertTrue(admit_curriculum(True,10000,10000,0.72,240,192))
        for values in ((False,10000,10000,.72,240,192),
                       (True,9000,10000,.72,240,192),
                       (True,10000,9000,.72,240,192),
                       (True,10000,10000,.80,240,192),
                       (True,10000,10000,.72,241,192),
                       (True,10000,10000,.72,240,257)):
            with self.subTest(values=values):
                self.assertFalse(admit_curriculum(*values))
        with self.assertRaises(ValueError):
            admit_curriculum(True,10000,10000,float("nan"),240,192)

    def test_apache2_oasst_chat_only_and_wrong_license_rejected(self):
        rows=self.fixture(4)
        rows=[
            x for x in rows if x[2]!="chat"
        ]
        rows.extend([
            ("human1","en","chat","oasst2-human-en","Apache-2.0",1),
            ("human2","de","chat","oasst2-human-de","Apache-2.0",2),
            ("bad1","en","chat","oasst2-human-en","unknown",3),
            ("bad2","en","math","oasst2-human-en","Apache-2.0",4),
        ])
        chosen=select_curriculum(rows,{"chat":2,"math":2})
        self.assertEqual({x[0] for x in chosen if x[2]=="chat"},{"human1","human2"})
        self.assertNotIn("bad2",{x[0] for x in chosen})

if __name__=="__main__":
    unittest.main()
