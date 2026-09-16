"""CPU-only, synthetic FP32 comparator fixtures; no model inference."""
import unittest
import math
from compare_fp32_holdout import compare

SAFE='a'*64
MID='b'*64
FINAL='c'*64


def fake(sha, tokens, loss):
    per=[{'batch':i+1,'tokens':2048,'nll_sum_fp32':loss*2048,
          'mean_nll_fp32':loss} for i in range(32)]
    return {'schema':1,'suite':'neral-nextgen-holdout-fp32-v1','checkpoint_sha256':sha,
            'checkpoint_processed_tokens':tokens,'architecture':'frontier350-v1','parameters':349825152,
            'read_only':True,'training_started':False,'promotion_allowed':False,'full_nis2_verified':False,
            'validation_tokens':65536,'seed':351337,'per_batch':per,'protocol_sha256':'d'*64,
            'tokenizer_sha256':'e'*64,'heldout_shard_sha256':[['validation.bin','f'*64]],
            'seq':512,'batch':4,'batches':32,'validation_loss_fp32':loss,
            'validation_perplexity_fp32':math.exp(loss)}


class Fp32ComparatorTests(unittest.TestCase):
    def fixture(self):
        return (fake(SAFE,1010003584,3.28224), fake(MID,1010249344,3.28232),
                fake(FINAL,1010495104,3.28221))

    def assess(self, rows=None):
        return compare(*(rows or self.fixture()),safe_sha=SAFE,mid_sha=MID,final_sha=FINAL)

    def test_detects_tiny_fp32_deltas_without_promoting(self):
        report=self.assess()
        self.assertGreater(report['candidates']['midpoint']['loss_delta_vs_safe'],0)
        self.assertLess(report['candidates']['final']['loss_delta_vs_safe'],0)
        self.assertFalse(report['promotion_allowed'])
        self.assertFalse(report['full_nis2_verified'])
        self.assertEqual(report['candidates']['final']['batches_with_lower_loss'],32)

    def test_checkpoint_and_protocol_tamper_rejected(self):
        rows=self.fixture();rows[1]['checkpoint_sha256']=FINAL
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            self.assess(rows)
        rows=self.fixture();rows[2]['protocol_sha256']='1'*64
        with self.assertRaisesRegex(ValueError,'protocol'):
            self.assess(rows)
        rows=self.fixture();rows[1]['heldout_shard_sha256']=[['changed.bin','1'*64]]
        with self.assertRaisesRegex(ValueError,'heldout'):
            self.assess(rows)

    def test_nonfinite_and_misreported_loss_rejected(self):
        for v in (float('nan'),float('inf'),'3.2',None,False):
            with self.subTest(value=v):
                rows=self.fixture();rows[1]['validation_loss_fp32']=v
                with self.assertRaises(ValueError):
                    self.assess(rows)
        rows=self.fixture();rows[1]['per_batch'][0]['nll_sum_fp32'] += 100
        with self.assertRaisesRegex(ValueError,'accounting|aggregate'):
            self.assess(rows)

    def test_missing_batches_or_unsafe_flag_rejected(self):
        rows=self.fixture();rows[2]['per_batch']=rows[2]['per_batch'][:-1]
        with self.assertRaisesRegex(ValueError,'incomplete'):
            self.assess(rows)
        rows=self.fixture();rows[1]['promotion_allowed']=True
        with self.assertRaisesRegex(ValueError,'safety'):
            self.assess(rows)

    def test_reused_or_wrong_checkpoint_sha_rejected(self):
        with self.assertRaisesRegex(ValueError,'distinct'):
            compare(*self.fixture(),safe_sha=SAFE,mid_sha=SAFE,final_sha=FINAL)
        rows=self.fixture();rows[1]['checkpoint_processed_tokens'] += 1
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            self.assess(rows)


if __name__=='__main__':
    unittest.main()
