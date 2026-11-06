import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.neral_math_review import check_math, choose_verified_math, build_review, restore_pinned_generator, git_blob_sha


def q(prompt, answer):
    return {"messages": [{"role":"system","content":"NERAL"},{"role":"user","content":prompt},{"role":"assistant","content":answer}]}


class MathReviewTests(unittest.TestCase):
    def test_all_eight_math_families_verified(self):
        data = [
            (0,q("Berechne 7 + 4 und erkläre den Rechenweg kurz.","7 + 4 = 11. Das Ergebnis ist 11.")),
            (1,q("Berechne 12 - 5. Zeige einen kurzen Rechenweg.","12 - 5 = 7.")),
            (2,q("Wie viel ist 3 mal 9? Begründe knapp.","3 · 9 = 27.")),
            (3,q("Löse die Gleichung 2x + (-3) = 11 nach x und zeige die Umformung.","2x = 14. Durch 2 teilen: x = 7.")),
            (4,q("Bestimme 25% von 120 und erkläre die Rechnung.","25% = 25/100. Daher 120 · 25/100 = 30.")),
            (5,q("Berechne 2/6 · 3 und kürze vollständig.","2/6 · 3 = 6/6 = 1.")),
            (6,q("Nutze F = m·a. Welche Kraft wirkt bei m = 4 kg und a = 5 m/s²? Zeige Formel und Ergebnis.","F = m·a = 4 kg · 5 m/s² = 20 N.")),
            (7,q("Setze die Zahlenfolge 2, 6, 10, 14, ... fort und nenne die Regel.","Die Differenz ist immer 4. Die nächste Zahl ist 18.")),
        ]
        for kind,item in data:
            with self.subTest(kind=kind): self.assertTrue(check_math(kind,item))
    def test_corrupted_results_rejected(self):
        for kind,p,a in [(0,'Berechne 7 + 4 und erkläre den Rechenweg kurz.','7 + 4 = 12.'),
                          (3,'Löse die Gleichung 2x + (-3) = 11 nach x und zeige die Umformung.','2x = 14. Durch 2 teilen: x = 8.'),
                          (5,'Berechne 2/6 · 3 und kürze vollständig.','2/6 · 3 = 6/6 = 2.'),
                          (7,'Setze die Zahlenfolge 2, 6, 10, 14, ... fort und nenne die Regel.','Die nächste Zahl ist 20.')]:
            with self.subTest(kind=kind):self.assertFalse(check_math(kind,q(p,a)))
    def test_unrecognised_prompt_rejected(self):
        self.assertFalse(check_math(0,q('Was ist 6+7?', '13')))
    def test_exact_generator_matching_and_others_blocked(self):
        match=q('Berechne 7 + 4 und erkläre den Rechenweg kurz.','7 + 4 = 11. Das Ergebnis ist 11.')
        rows=[match, q('Berechne 8 + 4 und erkläre den Rechenweg kurz.','8 + 4 = 12. Das Ergebnis ist 12.'),q('Hallo','Welt')]
        original={match['messages'][1]['content']:(0,match['messages'][2]['content'])}
        selected,stats=choose_verified_math(rows,original, max_rows=20)
        self.assertEqual(selected,[match]);self.assertEqual(stats['verified_math'],1)
        self.assertEqual(stats['excluded_no_exact_generator_match'],2)
    def test_review_hashes_and_zero_human_claim(self):
        e=q('Berechne 7 + 4 und erkläre den Rechenweg kurz.','7 + 4 = 11. Das Ergebnis ist 11.')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'math.jsonl'
            p.write_text(json.dumps(e,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf8')
            r=build_review(p, {'build_targeted_sft50_sources.py':'a'*64}, source_sha='b'*64)
            self.assertEqual(r['schema'],1)
            self.assertEqual(r['rows'][0]['line'],1)
            self.assertEqual(r['rows'][0]['row_sha256'],hashlib.sha256(p.read_text(encoding='utf8').strip().encode()).hexdigest())
            self.assertNotEqual(r['reviewer'],'Human')
            self.assertFalse(r['human_semantic_review'])
    def test_crlf_only_git_checkout_restored_and_pinned(self):
        original = b"import math\nVALUE = 10\n"
        expected = git_blob_sha(original)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pinned.py'
            p.write_bytes(original.replace(b'\n', b'\r\n'))
            restore_pinned_generator(p, expected)
            self.assertEqual(p.read_bytes(), original)
            # Repeating does not mutate already-correct checkout bytes.
            restore_pinned_generator(p, expected)
            self.assertEqual(p.read_bytes(), original)

    def test_unpinned_generator_is_not_repaired_or_accepted(self):
        original = b"import math\nVALUE = 10\n"
        expected = git_blob_sha(original)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pinned.py'
            bad = original.replace(b'VALUE = 10', b'VALUE = 20').replace(b'\n', b'\r\n')
            p.write_bytes(bad)
            with self.assertRaises(ValueError):
                restore_pinned_generator(p, expected)
            self.assertEqual(p.read_bytes(), bad)

    def test_no_other_files_or_outputs_in_review(self):
        e=q('Berechne 7 + 4 und erkläre den Rechenweg kurz.','7 + 4 = 11. Das Ergebnis ist 11.')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'math.jsonl';p.write_text(json.dumps(e)+'\n')
            report=build_review(p,{'build_targeted_sft50_sources.py':'a'*64},source_sha='b'*64)
            self.assertNotIn('Berechne 7',str(report))

if __name__=='__main__':unittest.main()
