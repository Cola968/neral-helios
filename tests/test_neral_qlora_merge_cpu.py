import importlib.util
import unittest
from scripts.neral_qlora_merge_peft import checkpoint_key,merge_tensor


class MergeRegression(unittest.TestCase):
    def test_text_adapter_maps_to_multimodal_checkpoint(self):
        self.assertEqual(checkpoint_key('base_model.model.model.layers.0.mlp.up_proj',{'model.language_model.layers.0.mlp.up_proj.weight':'shard'}),'model.language_model.layers.0.mlp.up_proj.weight')

    def test_missing_and_ambiguous_tensor_are_rejected(self):
        with self.assertRaises(ValueError): checkpoint_key('base_model.model.model.x',{})
        with self.assertRaises(ValueError): checkpoint_key('base_model.model.model.x',{'model.x.weight':'a','model.language_model.x.weight':'b'})

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch optional on CPU CI')
    def test_merge_preserves_base_and_applies_scaled_delta(self):
        import torch
        base=torch.ones(2,3,dtype=torch.bfloat16)
        a=torch.tensor([[1.,2.,3.]])
        b=torch.tensor([[1.],[2.]])
        merged=merge_tensor(base,a,b,2)
        self.assertTrue(torch.equal(merged,torch.tensor([[3,5,7],[5,9,13]],dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(base,torch.ones_like(base)))
        with self.assertRaises(ValueError): merge_tensor(base,a,b.T,2)
