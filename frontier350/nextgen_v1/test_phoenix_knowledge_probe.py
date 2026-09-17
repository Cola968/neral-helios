"""Synthetic/contract tests: no Torch, model, local user data or GPU."""
import math
import unittest

import phoenix_knowledge_probe as probe


class PublicKnowledgeProbeContracts(unittest.TestCase):
    def test_twelve_public_facts_are_balanced(self):
        self.assertTrue(probe.validate_cases())
        self.assertEqual(len(probe.CASES), 12)
        self.assertEqual(sum(row[0] == 'de' for row in probe.CASES), 6)

    def test_duplicates_or_bad_alternatives_are_rejected(self):
        cases = list(probe.CASES)
        cases[1] = cases[0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            probe.validate_cases(tuple(cases))
        cases = list(probe.CASES)
        cases[0] = ('en', cases[0][1], '4', ' 7')
        with self.assertRaisesRegex(ValueError, 'malformed'):
            probe.validate_cases(tuple(cases))

    def test_no_format_is_substituted_for_another(self):
        self.assertEqual(probe.prompt_for('en', 'Hello?', 'chat'), 'User: Hello?\nAssistant:')
        self.assertEqual(probe.prompt_for('en', 'Hello?', 'qa'), 'Question: Hello?\nAnswer:')
        self.assertEqual(probe.prompt_for('de', 'Hallo?', 'qa'), 'Frage: Hallo?\nAntwort:')
        with self.assertRaises(ValueError):
            probe.prompt_for('de', 'Hallo?', 'unknown')

    def test_finite_comparisons_and_ties(self):
        self.assertEqual(probe.normalized_pair(-1.0, -2.0), (1, 1.0))
        self.assertEqual(probe.normalized_pair(-4.0, -2.0), (0, -2.0))
        self.assertEqual(probe.normalized_pair(-2.0, -2.0), (-1, 0.0))
        for invalid in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                probe.normalized_pair(invalid, -2.0)


if __name__ == '__main__':
    unittest.main()
