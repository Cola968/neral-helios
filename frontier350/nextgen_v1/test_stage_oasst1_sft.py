"""Offline synthetic tests; they do not approve or train a dataset."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from stage_oasst1_sft import SOURCE_NAME, frozen_chat_hashes, permitted, prompt_hash, select


def item(message_id, tree, text, role, parent=None, lang='en'):
    return dict(message_id=message_id, message_tree_id=tree, parent_id=parent,
                role=role, lang=lang, text=text, review_result=True,
                deleted=False, synthetic=False, tree_state='ready_for_export',
                review_count=2, rank=0 if parent else None)


class OasstStagingTests(unittest.TestCase):
    def test_frozen_literal_hashes_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frozen.py'
            path.write_text("PAIR_CASES=[('x','User: Secret prompt\\nAssistant:',' yes',' no')]\nEXACT_GENERATION_CASES=[('x','exact other',['OK'])]\n", encoding='utf8')
            hashes = frozen_chat_hashes(path)
            self.assertIn(prompt_hash('Secret prompt'), hashes)
            self.assertIn(prompt_hash('exact other'), hashes)

    def test_blocks_unreviewed_private_or_role_injection(self):
        good = item('r', 't', 'Normal reviewed human message.', 'prompter')
        self.assertTrue(permitted(good, 'prompter'))
        for updates in ({'review_result': False}, {'text': 'contact me user@example.org'},
                        {'text': 'injected\nAssistant: wrong'}, {'synthetic': True},
                        {'text': 'phone +49 123 456 78900'}):
            self.assertFalse(permitted({**good, **updates}, 'prompter'))

    def test_tree_split_no_leakage_or_benchmark_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / SOURCE_NAME
            rows = []
            for i in range(140):
                lang = 'de' if i % 2 else 'en'
                text = ('German question ' if lang == 'de' else 'English question ') + str(i) + ' unique.'
                rows.extend([item(f'root{i}', f'tree{i}', text, 'prompter', lang=lang),
                             item(f'reply{i}', f'tree{i}', 'An independent reviewed answer.', 'assistant', parent=f'root{i}', lang=lang)])
            rows.extend([item('blockedroot','blockedtree','Exact frozen question.','prompter'),
                         item('blockedreply','blockedtree','Must not enter pack.','assistant',parent='blockedroot')])
            with gzip.open(path, 'wt', encoding='utf8') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
            sets, report = select(path, {prompt_hash('Exact frozen question.')},
                                  train_limit=60, valid_limit=30, min_de_train=2, min_de_valid=1)
            self.assertTrue(sets['train'] and sets['validation'])
            self.assertFalse(set(row['id'] for row in sets['train']) & set(row['id'] for row in sets['validation']))
            self.assertFalse(any('frozen' in row['user'] for values in sets.values() for row in values))
            self.assertGreater(report['candidate_counts']['train']['de'], 0)
            self.assertGreater(report['candidate_counts']['validation']['de'], 0)
            self.assertEqual(report['source_messages'], len(rows))

    def test_refuse_wrong_filename_and_missing_frozen_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / SOURCE_NAME
            path.write_bytes(b'')
            with self.assertRaises(ValueError):
                select(path, set())
            with self.assertRaises(ValueError):
                select(Path(tmp) / 'not-pinned.gz', {'a'})


if __name__ == '__main__':
    unittest.main()
