"""CPU-only checks of the actual Qwen completion-mask audit."""
import unittest
from scripts.neral_v3_token_audit import summarize_masks, PINNED_DATA


class TokenBoundaryTests(unittest.TestCase):
    def test_masks_include_context_but_train_only_completion(self):
        records=[
            {'input_ids':[1,2,3,4,5], 'completion_mask':[0,0,0,1,1]},
            {'input_ids':[1,2,3], 'completion_mask':[0,1,1]},
        ]
        s=summarize_masks(records,512)
        self.assertEqual(s['count'],2)
        self.assertEqual(s['tokens_total'],8)
        self.assertEqual(s['completion_tokens'],4)
        self.assertEqual(s['max_tokens'],5)
        self.assertEqual(s['overflow'],0)

    def test_context_tokens_must_not_receive_loss(self):
        with self.assertRaises(ValueError):
            summarize_masks([{'input_ids':[1,2,3], 'completion_mask':[0,1,0]}],512)

    def test_no_missing_completion(self):
        with self.assertRaises(ValueError):
            summarize_masks([{'input_ids':[1,2], 'completion_mask':[0,0]}],512)

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            summarize_masks([{'input_ids':list(range(513)),
                              'completion_mask':[0]*3+[1]*510}],512)

    def test_source_is_pinned_to_exact_split_run(self):
        self.assertEqual(PINNED_DATA['train']['count'],320)
        self.assertEqual(PINNED_DATA['validation']['count'],64)
        for v in PINNED_DATA.values():
            self.assertEqual(len(v['sha256']),64)
            self.assertEqual(len(v['review_sha256']),64)


if __name__=='__main__':
    unittest.main()
