import importlib.util
import tempfile
import unittest
from pathlib import Path
from scripts.neral_qlora_train_peft import check_local_inputs

class PeftInputRegression(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"),"Optional PyTorch CPU check")
    def test_large_frozen_embeddings_keep_original_dtype(self):
        import torch
        from scripts.neral_qlora_train_peft import prepare_memory_bounded
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed=torch.nn.Embedding(8,4,dtype=torch.bfloat16)
                self.head=torch.nn.Linear(4,8,bias=False,dtype=torch.bfloat16)
                self.norm=torch.nn.LayerNorm(4,dtype=torch.bfloat16)
            def get_input_embeddings(self): return self.embed
            def get_output_embeddings(self): return self.head
        def prepare(model,**kwargs):
            self.assertEqual(kwargs["gradient_checkpointing_kwargs"],{"use_reentrant":False})
            for param in model.parameters(): param.requires_grad_(False)
            return model.float()
        model=prepare_memory_bounded(Tiny(),prepare,min_elements=1)
        self.assertEqual(model.embed.weight.dtype,torch.bfloat16)
        self.assertEqual(model.head.weight.dtype,torch.bfloat16)
        self.assertEqual(model.norm.weight.dtype,torch.float32)
        self.assertFalse(model.head.weight.requires_grad)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"Optional PyTorch CPU check")
    def test_selected_logits_match_full_causal_loss_and_gradients(self):
        import torch
        from types import SimpleNamespace
        from scripts.neral_qlora_train_peft import answer_only_loss
        torch.manual_seed(19)
        hidden=torch.randn(1,7,4,requires_grad=True)
        weight=torch.randn(5,4,requires_grad=True)
        labels=torch.tensor([[-100,-100,-100,2,4,1,-100]])
        full=hidden@weight.T
        reference=torch.nn.functional.cross_entropy(full[:,:-1].reshape(-1,5),labels[:,1:].reshape(-1),ignore_index=-100)
        ref_grads=torch.autograd.grad(reference,(hidden,weight),retain_graph=True)
        def model(*,logits_to_keep,**kwargs):
            return SimpleNamespace(logits=hidden[:,logits_to_keep]@weight.T)
        loss,_=answer_only_loss(model,{"input_ids":torch.zeros_like(labels),"labels":labels})
        grads=torch.autograd.grad(loss,(hidden,weight),retain_graph=True)
        self.assertTrue(torch.allclose(loss,reference,atol=1e-6))
        for actual,expected in zip(grads,ref_grads): self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        scaled,_=answer_only_loss(model,{"labels":labels},num_items_in_batch=6)
        self.assertTrue(torch.allclose(scaled,reference/2,atol=1e-6))
        with self.assertRaises(ValueError): answer_only_loss(model,{"labels":torch.full_like(labels,-100)})

    def test_smoke_has_no_zero_lr_warmup(self):
        from scripts.neral_qlora_train_peft import optimizer_schedule
        self.assertEqual(optimizer_schedule(720,1,1),{"warmup_steps":0})
        self.assertEqual(optimizer_schedule(720,1,0),{"warmup_steps":3})

    def test_mapping_chat_template_masks_prefix(self):
        from scripts.neral_qlora_train import encode_answer_only
        class MappingTokenizer:
            def apply_chat_template(self,messages,**kwargs):
                return {'input_ids':[1,2] if kwargs['add_generation_prompt'] else [1,2,3,4]}
        rows=[{'messages':[{'role':'system','content':'s'},{'role':'user','content':'u'},{'role':'assistant','content':'a'}]}]
        encoded=encode_answer_only(rows,MappingTokenizer(),1024)
        self.assertEqual(encoded[0]['labels'],[-100,-100,3,4])
    def test_cpu_check_only_requires_no_model_or_gpu(self):
        import contextlib,io,json
        from unittest.mock import patch
        from scripts.neral_qlora_train_peft import main
        stdout=io.StringIO()
        with patch("sys.argv",["trainer","--train","train","--holdout","holdout","--out","out","--check-only"]),patch("scripts.neral_qlora_train_peft.check_local_inputs",return_value=[{}]*720),contextlib.redirect_stdout(stdout):
            self.assertEqual(main(),0)
        self.assertEqual(json.loads(stdout.getvalue())["rows"],720)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(FileExistsError):
                check_local_inputs(root/'train',root/'holdout',root,steps=1,epochs=1,context=1024)

    def test_holdout_cannot_be_training_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(ValueError):
                check_local_inputs(root/'train',root/'train',root/'out',steps=1,epochs=1,context=1024)

    def test_unbounded_smoke_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(ValueError):
                check_local_inputs(root/'train',root/'holdout',root/'out',steps=3,epochs=1,context=1024)

if __name__=='__main__': unittest.main()
