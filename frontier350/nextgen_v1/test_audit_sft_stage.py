"""Offline fixtures; actual datasets never enter CI or logs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_sft_stage import run, similar


class StageReviewTests(unittest.TestCase):
    def test_similarity_requires_sufficient_overlap(self):
        self.assertEqual(similar({'one','two'}, {'one','two'}), 0)
        self.assertEqual(similar(set('a b c d e f'.split()), set('a b c d e f'.split())), 1)

    def test_flags_without_publishing_text_or_approving(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            benchmark = folder / 'b.py'
            benchmark.write_text("PAIR_CASES=[('x','User: Never train on this.\\nAssistant:',' good',' bad')]\nEXACT_GENERATION_CASES=[('x','Only YES',['YES'])]\n", encoding='utf8')
            train = folder / 'train.jsonl'
            valid = folder / 'validation.jsonl'
            first = {'id':'train1','source_id':'fixture','user':'Where can I find the first yellow tree near the park?', 'assistant':'Contact me at sample@example.org'}
            second = {'id':'val1','source_id':'fixture','user':'Where can I find the first yellow tree near the park?', 'assistant':'Do not copy the example.'}
            train.write_text(json.dumps(first)+'\n', encoding='utf8')
            # The same exact prompt must be rejected even before fuzzy scoring.
            valid.write_text(json.dumps(second)+'\n', encoding='utf8')
            def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'duplicate candidate prompt'):
                run(train,valid,benchmark,sha(train),sha(valid))
            second['user']='Where do I find the first yellow tree near the park?'
            valid.write_text(json.dumps(second)+'\n', encoding='utf8')
            result=run(train,valid,benchmark,sha(train),sha(valid))
            self.assertEqual(result['schema'],2)
            self.assertFalse(result['approved_for_training'])
            self.assertFalse(result['training_started'])
            self.assertEqual(result['candidate_rows_requiring_privacy_or_format_review'],1)
            self.assertNotIn('sample@example.org',json.dumps(result))
            with self.assertRaisesRegex(ValueError,'staging split changed'):
                run(train,valid,benchmark,'0'*64,sha(valid))

    def test_full_dialogue_similarity_catches_copied_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            benchmark=folder/'b.py'
            benchmark.write_text("PAIR_CASES=[]\nEXACT_GENERATION_CASES=[]\n",encoding='utf8')
            train=folder/'train.jsonl'; valid=folder/'validation.jsonl'
            shared='alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'
            a={'id':'t','source_id':'fixture','user':'How does a turbine work in simple terms?', 'assistant':shared}
            b={'id':'v','source_id':'fixture','user':'Explain photosynthesis to a beginner please?', 'assistant':shared}
            train.write_text(json.dumps(a)+'\n',encoding='utf8')
            valid.write_text(json.dumps(b)+'\n',encoding='utf8')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            result=run(train,valid,benchmark,sha(train),sha(valid))
            self.assertEqual(result['cross_split_near_duplicate_pairs'],1)
            hit=result['near_duplicate_ids_no_text'][0]
            self.assertLess(hit['prompt_jaccard'],0.84)
            self.assertGreaterEqual(hit['dialogue_jaccard'],0.84)


if __name__=='__main__':
    unittest.main()
