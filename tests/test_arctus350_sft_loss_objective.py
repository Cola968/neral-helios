"""Arctus350 opt-in per-example SFT objective: deterministic CPU proof, NO weights.

The legacy token mean must remain byte-for-byte numerically equivalent to
the existing cross-entropy; the new objective must genuinely affect gradients.
"""
import unittest
import torch
import torch.nn.functional as F

from frontier350.sft_loss350 import assistant_loss_from_logits


class PerExampleSFTLossTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([
            [[2., 0., 1., -1.], [.2, 2., -.3, 0.], [1., -.5, 2., 0.], [1., 0., -.5, 2.]],
            [[-.1, 2., .4, .8], [.1, 0., 1., 3.], [1., 2., 3., 0.], [0., 0., 0., 0.]],
        ], dtype=torch.float32, requires_grad=True)
        self.targets = torch.tensor([
            [-100, 1, 2, 1],
            [3, -100, -100, -100],
        ], dtype=torch.long)

    def test_default_token_weighting_exactly_matches_legacy(self):
        masked = self.targets != -100
        selected = self.logits[masked]
        targets = self.targets[masked]
        losses = F.cross_entropy(selected.float(), targets, reduction="none")
        weights = torch.where(targets==1, 1.5, 1.)
        expected = (losses*weights).sum()/weights.sum()
        actual = assistant_loss_from_logits(self.logits,self.targets, eos_id=1, eos_weight=1.5)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-7)

    def test_per_example_mean_equalizes_short_and_long_answers(self):
        y=self.targets
        per=F.cross_entropy(self.logits[y!=-100],y[y!=-100],reduction="none")
        # row 1 has three assistant tokens and two EOS-weighted targets.
        first=(per[:3]*torch.tensor([1.5,1.,1.5])).sum()/4.
        second=per[3]
        expected=(first+second)/2.
        actual=assistant_loss_from_logits(self.logits,y,eos_id=1,eos_weight=1.5,aggregation="per-example")
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-7)
        self.assertGreater(abs(float(actual.detach()-assistant_loss_from_logits(
            self.logits,y,eos_id=1,eos_weight=1.5,aggregation="token"))),0.02)
        actual.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertEqual(float(self.logits.grad[1,1].abs().sum()),0.0)

    def test_single_example_has_identical_objective(self):
        for width in (1,3):
            targets=self.targets[:1,:width].clone()
            if width==1:targets[0,0]=1
            a=assistant_loss_from_logits(self.logits[:1,:width],targets,eos_id=1,eos_weight=2.,aggregation="token")
            b=assistant_loss_from_logits(self.logits[:1,:width],targets,eos_id=1,eos_weight=2.,aggregation="per-example")
            torch.testing.assert_close(a,b,rtol=0,atol=1e-7)

    def test_any_empty_assistant_row_is_rejected_in_per_example_mode(self):
        y=self.targets.clone()
        y[1,:]=-100
        with self.assertRaisesRegex(ValueError,"no assistant targets"):
            assistant_loss_from_logits(self.logits,y,eos_id=1,eos_weight=1.5,aggregation="per-example")
        # Legacy mode is intentionally unchanged: only completely empty batches fail.
        val=assistant_loss_from_logits(self.logits,y,eos_id=1,eos_weight=1.5,aggregation="token")
        self.assertTrue(torch.isfinite(val))
        with self.assertRaises(ValueError):
            assistant_loss_from_logits(self.logits,torch.full_like(y,-100),eos_id=1,eos_weight=1.5)

    def test_invalid_modes_shapes_and_weights_are_rejected(self):
        for mode in ("per-token","sample","sum",None):
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                assistant_loss_from_logits(self.logits,self.targets,1,1.5,aggregation=mode)
        for weight in (0.,-1.,float("nan"),float("inf"),True):
            with self.subTest(weight=weight),self.assertRaises(ValueError):
                assistant_loss_from_logits(self.logits,self.targets,1,weight)
        with self.assertRaises(ValueError):
            assistant_loss_from_logits(self.logits[0],self.targets,1,1.5)
        with self.assertRaises(ValueError):
            assistant_loss_from_logits(self.logits,self.targets.float(),1,1.5)
        with self.assertRaises(ValueError):
            assistant_loss_from_logits(self.logits,self.targets,9,1.5)

if __name__=="__main__":unittest.main()
