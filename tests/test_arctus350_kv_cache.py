"""Arctus350 grouped-query KV-cache contract against real unfused tiny architecture.

Tiny randomly initialized native architecture is used for CPU equivalence;
this proves numerical parity, NOT real checkpoint speed or model quality.
"""
from __future__ import annotations
import unittest
import torch
from frontier350.model350 import Neral350Config, Neral350M


def tiny():
    return Neral350Config(
        vocab_size=96,d_model=64,n_layers=2,n_heads=4,n_kv_heads=2,
        ffn_hidden=128,max_seq_len=32,dropout=0.0,
    )


class NativeArctusKVCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261029)
        self.model=Neral350M(tiny()).eval()

    def test_prefill_and_each_decode_token_identical_to_full_forward(self):
        prompt=torch.randint(0,96,(2,12))
        with torch.inference_mode():
            expected,_=self.model(prompt)
            logits,state=self.model.forward_cached(prompt[:,:5])
            self.assertTrue(torch.allclose(logits,expected[:,:5],atol=1e-5,rtol=1e-5))
            for i in range(5,12):
                self.assertEqual(len(state),2)
                for kv in state:
                    self.assertEqual(tuple(kv[0].shape),(2,2,i,16))
                    self.assertEqual(tuple(kv[1].shape),(2,2,i,16))
                logits,state=self.model.forward_cached(prompt[:,i:i+1],state)
                self.assertTrue(torch.allclose(logits[:,-1,:],expected[:,i,:],atol=2e-5,rtol=2e-5),
                    f"native parity diverged at position {i}")

    def test_same_weights_no_training_mutation_and_autograd_still_works(self):
        before={name:weight.clone() for name,weight in self.model.state_dict().items()}
        with torch.inference_mode():
            _,cache=self.model.forward_cached(torch.randint(0,96,(1,9)))
        self.assertEqual(len(before),len(self.model.state_dict()))
        self.assertEqual(set(before),set(self.model.state_dict()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in self.model.state_dict().items()))
        self.model.train()
        tokens=torch.randint(0,96,(1,7))
        _,loss=self.model(tokens,tokens)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in self.model.parameters()))
        with self.assertRaises(ValueError):
            self.model.forward_cached(tokens)

    def test_rejects_reused_or_malformed_cache_and_overflow(self):
        with torch.inference_mode():
            _,state=self.model.forward_cached(torch.randint(0,96,(1,6)))
            wrong=[(x.clone(),y.clone()) for x,y in state]
            wrong[0]=(wrong[0][0][:,:,:-1,:],wrong[0][1])
            with self.assertRaises(ValueError):
                self.model.forward_cached(torch.randint(0,96,(1,1)),wrong)
            with self.assertRaises(ValueError):
                self.model.forward_cached(torch.randint(0,96,(1,2)),state)
            with self.assertRaises(ValueError):
                self.model.forward_cached(torch.randint(0,96,(2,1)),state)
            with self.assertRaises(ValueError):
                self.model.forward_cached(torch.randint(0,96,(1,1)),[])
            with self.assertRaises(ValueError):
                self.model.forward_cached(torch.randint(0,96,(1,32)),state)


    def test_opt_in_cli_greedy_matches_uncached_chat_response(self):
        from types import SimpleNamespace
        from frontier350.chat_cli350 import generate
        class TinyTokenizer:
            def token_to_id(self,name):
                return {"<|bos|>":1,"<|eos|>":2,"<|pad|>":3,"<|sep|>":4}.get(name)
            def encode(self,text):
                return SimpleNamespace(ids=[5+ord(c)%91 for c in text[::2]])
            def decode(self,ids,skip_special_tokens=True):
                return ".".join(map(str,ids))
        tok=TinyTokenizer()
        history=[("user","Hallo Arctus")]
        with torch.inference_mode():
            old,old_h=generate(self.model,tok,history,max_new=10,max_context=42,
                               temperature=0,top_p=1,repetition_penalty=1.08,use_kv_cache=False)
            new,new_h=generate(self.model,tok,history,max_new=10,max_context=42,
                               temperature=0,top_p=1,repetition_penalty=1.08,use_kv_cache=True)
        self.assertEqual(old,new)
        self.assertEqual(old_h,new_h)

    def test_repeated_greedy_next_token_matches_full_recompute(self):
        seed=torch.randint(0,96,(1,8))
        with torch.inference_mode():
            legacy=seed.clone()
            cached=seed.clone()
            next_logits,state=self.model.forward_cached(cached)
            for _ in range(10):
                expected,_=self.model(legacy)
                a=expected[:,-1,:].argmax(-1,keepdim=True)
                b=next_logits[:,-1,:].argmax(-1,keepdim=True)
                self.assertTrue(torch.equal(a,b))
                legacy=torch.cat((legacy,a),dim=1)
                cached=torch.cat((cached,b),dim=1)
                if cached.shape[1] < 18:
                    next_logits,state=self.model.forward_cached(b,state)
            self.assertTrue(torch.equal(legacy,cached))


if __name__ == "__main__":
    unittest.main()
