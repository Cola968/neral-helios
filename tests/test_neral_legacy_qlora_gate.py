"""CPU-only tests of the NeralTalk -> Qwen trainer admission boundary."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.neral_legacy_qlora_gate import (
    fingerprint, check_qwen_template, plan_integration, publish_integration,
    validate_source_receipt,
)

class FakeTokenizer:
    """Simple deterministic chat serializer; never imports transformers/torch."""
    def apply_chat_template(self, messages, *, chat_template, tokenize, add_generation_prompt):
        text = ''.join(f'<{m["role"]}>\n{m["content"]}\n' for m in messages)
        if add_generation_prompt:
            text += '<assistant>\n'
        return [ord(c) for c in text]


def example(question='Wie funktioniert Python?', answer='Die Funktion gibt den Wert zurueck.'):
    return {'messages': [
        {'role': 'system', 'content': 'NERAL ist ein transparenter Assistent.'},
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': answer},
    ]}

class LegacyQwenAdmissionTests(unittest.TestCase):
    def test_fingerprint_normalizes_case_spaces_and_unicode(self):
        self.assertEqual(fingerprint('  GRÜßE   Dich '), fingerprint('grüße dich'))

    def test_model_tokenizer_prefix_boundary_and_overlength(self):
        tok = FakeTokenizer()
        self.assertGreater(check_qwen_template(example(), tok, 'test', 1024), 0)
        self.assertIsNone(check_qwen_template(example(), tok, 'test', 10))

    def test_holdout_and_prior_trainer_duplicates_are_excluded(self):
        examples = [example('Sicher?'), example('Neue Frage')]
        plan = plan_integration(examples, FakeTokenizer(), 'template', max_length=1024,
                                holdout_prompts={fingerprint(' SICHER? ')},
                                previous_prompts={fingerprint('neue frage')})
        self.assertEqual(plan['counts']['heldout_or_prior_overlap'], 2)
        self.assertEqual(len(plan['eligible']), 0)
        self.assertFalse(plan['approved_for_training'])

    def test_unknown_source_or_mismatched_receipt_refuses(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / 'dataset.jsonl'
            source.write_text(json.dumps(example(), ensure_ascii=False)+'\n', encoding='utf8')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            validate_source_receipt(source, {'sha256': digest, 'rows': 1})
            with self.assertRaises(ValueError):
                validate_source_receipt(source, {'sha256': '0'*64, 'rows': 1})
            with self.assertRaises(ValueError):
                validate_source_receipt(source, {'sha256': digest, 'rows': 2})

    def test_selection_is_stable_and_keeps_private_data_local(self):
        a, b = example('Aufgabe A'), example('Aufgabe B')
        p = plan_integration([a, b], FakeTokenizer(), 'template', max_length=1024,
                             holdout_prompts=set(), previous_prompts=set())
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)/'new'
            report = publish_integration(folder, p, source_sha='a'*64,
                                         source_split_shas={'train':'b'*64,'val':'c'*64,'test':'d'*64})
            self.assertEqual(report['status'], 'INTEGRATED_REVIEW_REQUIRED')
            self.assertEqual(report['eligible_rows'], 2)
            self.assertFalse(report['approved_for_training'])
            self.assertNotIn('Aufgabe A', json.dumps(report, ensure_ascii=False))
            self.assertNotIn('Aufgabe B', json.dumps(report, ensure_ascii=False))
            self.assertTrue((folder/'train_candidate.jsonl').exists())
            with self.assertRaises(FileExistsError):
                publish_integration(folder,p,source_sha='a'*64,source_split_shas={})

    def test_repeated_prompts_or_answers_are_not_mistaken_for_review(self):
        rows = [example('Einmal'), example(' EINMAL ', 'Andere Antwort.'),
                example('Frage neu', 'Die Funktion gibt den Wert zurueck.')]
        p = plan_integration(rows,FakeTokenizer(),'template',max_length=1024,
                             holdout_prompts=set(), previous_prompts=set())
        self.assertEqual(p['counts']['duplicate_prompts'], 1)
        self.assertEqual(p['counts']['repeated_answer_pairs'], 1)
        self.assertEqual(len(p['eligible']), 2)
        self.assertFalse(p['approved_for_training'])

    def test_empty_output_not_promoted(self):
        p = plan_integration([example(answer='A'*2000)],FakeTokenizer(),'template',max_length=512,
                             holdout_prompts=set(),previous_prompts=set())
        self.assertEqual(len(p['eligible']),0)
        self.assertEqual(p['counts']['over_token_limit'],1)
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                publish_integration(Path(tmp)/'output',p,source_sha='a'*64,source_split_shas={})

    def test_no_automatic_training_approval_even_full_token_coverage(self):
        p = plan_integration([example('1+1?', '2 ist korrekt.')],FakeTokenizer(),'template',
                             max_length=1024,holdout_prompts=set(),previous_prompts=set())
        self.assertEqual(p['counts']['eligible'],1)
        self.assertTrue(p['format_compatible'])
        self.assertFalse(p['approved_for_training'])
        self.assertTrue(p['semantic_review_required'])

if __name__ == '__main__': unittest.main()
