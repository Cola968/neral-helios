"""Reference-mode tests using only the repository's public tiny corpus; no model/GPU."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from helios_reference_knowledge import ReferenceStore, normalized, _source

CORPUS = Path(__file__).with_name('helios_reference_facts_v1.json')


class ReferenceKnowledgeContracts(unittest.TestCase):
    def test_bundled_corpus_and_english_german_exact_retrieval(self):
        store = ReferenceStore(CORPUS)
        self.assertEqual(store.fact_count, 14)
        self.assertEqual(store.verified_on, '2026-09-17')
        self.assertEqual(len(store.sha256), 64)
        de = store.lookup('   WIE VIELE PLANETEN HAT UNSER SONNENSYSTEM   ')
        en = store.lookup('How many planets are in our solar system?')
        self.assertEqual(de['answer'], 'Unser Sonnensystem hat acht Planeten.')
        self.assertEqual(en['answer'], 'Our solar system has eight planets.')
        self.assertEqual(de['status'], 'exact_reference_match')
        self.assertFalse(de['model_generated'])
        self.assertTrue(de['source']['url'].startswith('https://science.nasa.gov/'))

    def test_abstains_on_unknown_and_related_but_not_exact_questions(self):
        store = ReferenceStore(CORPUS)
        for question in ('Was ist das Wetter heute?', 'Wie viele Monde hat Jupiter?',
                         'Wie viele Planeten hatte unser Sonnensystem im Jahr 1500?'):
            result = store.lookup(question)
            self.assertEqual(result['status'], 'abstained_no_exact_reference')
            self.assertIsNone(result['source'])
            self.assertFalse(result['model_generated'])

    def test_normalization_does_not_turn_unrelated_text_into_alias(self):
        self.assertEqual(normalized('  WIE viele  Planeten?  '), 'wie viele planeten')
        self.assertNotEqual(normalized('Wie viele Planeten gibt es?'),
                            normalized('Wie viele Planeten gab es?'))

    def test_duplicate_alias_or_untrusted_source_rejected(self):
        original = json.loads(CORPUS.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'facts.json'
            bad = json.loads(json.dumps(original))
            bad['facts'][1]['questions']['en'][0] = bad['facts'][0]['questions']['en'][0]
            path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                ReferenceStore(path)
            bad = json.loads(json.dumps(original))
            bad['facts'][0]['source_url'] = 'https://attacker.example/lie'
            path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'allowed HTTPS'):
                ReferenceStore(path)
            bad = json.loads(json.dumps(original))
            bad['facts'][0]['answers']['de'] = ''
            path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'answer'):
                ReferenceStore(path)

    def test_source_url_rejects_lookalike_credentials_and_redirect_payloads(self):
        self.assertFalse(_source('https://www.nist.gov.attacker.example/a'))
        self.assertFalse(_source('https://www.nist.gov@attacker.example/a'))
        self.assertFalse(_source('http://science.nasa.gov/a'))
        self.assertFalse(_source('https://science.nasa.gov/a?next=evil'))
        self.assertTrue(_source('https://science.nasa.gov/solar-system/planets/'))

    def test_refuses_symlink_to_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            link = Path(temp) / 'linked.json'
            link.symlink_to(CORPUS)
            with self.assertRaises(ValueError):
                ReferenceStore(link)


if __name__ == '__main__':
    unittest.main()
