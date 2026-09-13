import copy
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier350'))
from model350 import Neral350Config, Neral350M
from protection_v9 import configure_trainable
from speed_execution_v5 import enable_fused_execution, execution_groups, backward_update, transfer_update
from train350_compiled import mask_budget_targets, update_token_counts


class SpeedEquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(731)
        self.cfg = Neral350Config(vocab_size=32, d_model=16, n_layers=5,
            n_heads=4, n_kv_heads=2, ffn_hidden=24, max_seq_len=16)
        self.model = Neral350M(self.cfg)
        configure_trainable(self.model, 4)

    def test_fusion_preserves_logits_gradients_optimizer_and_frozen_weights(self):
        candidate = copy.deepcopy(self.model)
        before = {n: id(p) for n, p in candidate.named_parameters()}
        enable_fused_execution(candidate)
        self.assertEqual(before, {n: id(p) for n, p in candidate.named_parameters()})
        self.assertEqual(list(self.model.state_dict()), list(candidate.state_dict()))
        frozen = {n: p.detach().clone() for n, p in candidate.named_parameters() if not p.requires_grad}
        optimizers = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
            lr=2.6e-5, betas=(.9,.95), weight_decay=.1) for m in (self.model,candidate)]
        for _ in range(3):
            x = torch.randint(0,32,(2,8)); y = torch.randint(0,32,(2,8))
            outputs=[]
            for model, optimizer in zip((self.model,candidate), optimizers):
                optimizer.zero_grad(set_to_none=True)
                logits, loss = model(x,y); loss.backward(); outputs.append((logits,loss))
            torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=2e-6, rtol=2e-5)
            for (n,p),(m,q) in zip(self.model.named_parameters(),candidate.named_parameters()):
                self.assertEqual(n,m)
                if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)
            for model,optimizer in zip((self.model,candidate),optimizers):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step()
            for p,q in zip(self.model.parameters(),candidate.parameters()):
                torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-4)
        for n,p in candidate.named_parameters():
            if n in frozen:self.assertTrue(torch.equal(p,frozen[n]))
        reloaded=Neral350M(self.cfg);reloaded.load_state_dict(candidate.state_dict(),strict=True)
        self.assertIs(reloaded.lm_head.weight,reloaded.tok_emb.weight)

    def test_replay_mixed_packs_and_tail_preserve_weighted_objective(self):
        # Includes exactly replay-only, partially filled main row, and full updates.
        for budget in (48,49,71,192):
            counts=update_token_counts(budget,16,12)
            batches=[]
            for count,rows in [(c,2) for c in counts[:3]] + ([(sum(counts[3:]),2*len(counts[3:]))] if counts[3:] else []):
                x=torch.randint(0,32,(rows,8));y=torch.randint(0,32,(rows,8))
                batches.append((x,mask_budget_targets(y,count)))
            denominator=sum(counts[3:])+1.78*sum(counts[:3])
            reference=None
            for pack,fuse in ((9,False),(9,True),(10,True),(11,True),(12,True)):
                m=copy.deepcopy(self.model)
                if fuse:enable_fused_execution(m)
                staged=transfer_update(batches,'cpu')
                for a,b in zip(batches,staged):
                    self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[1],b[1]))
                loss,flags=backward_update(m,execution_groups(staged,counts,2,1.78,pack,fuse),denominator,'cpu')
                self.assertTrue(all(flags))
                grads=torch.cat([p.grad.flatten() for p in m.parameters() if p.requires_grad])
                if reference is None:reference=(loss,grads)
                torch.testing.assert_close(loss,reference[0],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(grads,reference[1],atol=2e-6,rtol=2e-4)

    def test_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):list(execution_groups([],[],4,1.5,13,True))
        with self.assertRaises(ValueError):list(execution_groups([], [1,1,1],4,1.5,10,False))


if __name__=='__main__':unittest.main()
