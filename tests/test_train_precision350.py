"""Train precision policy tests; pure mocks so GitHub CI needs no torch/GPU."""
from __future__ import annotations
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from frontier350.train_precision350 import (
    validate_precision_mode, forward_with_amp, checkpoint_model_state,
)

class FakeTensor:
    def __init__(self,dtype):
        self.dtype=dtype
        self.device=SimpleNamespace(type="cuda")
        self.moves=[]
    def detach(self): return self
    def to(self,dtype):
        self.moves.append(("to",dtype))
        return self
    def cpu(self):
        self.moves.append(("cpu",))
        return self

class FakeTorch:
    float32="float32"
    bfloat16="bfloat16"
    def __init__(self):self.calls=[]
    @contextmanager
    def autocast(self,device_type,dtype,enabled):
        self.calls.append((device_type,dtype,enabled))
        yield

class Model:
    def __init__(self,dtype="float32",device="cuda"):
        self.param=FakeTensor(dtype)
        self.param.device.type=device
        self.calls=[]
    def parameters(self): yield self.param
    def __call__(self,*args):
        self.calls.append(args)
        return ("logits","loss")
    def state_dict(self):return {"weights":FakeTensor(self.param.dtype)}

class PrecisionPolicyTests(unittest.TestCase):
    def test_opt_in_modes_are_strict(self):
        self.assertEqual(validate_precision_mode("legacy-bf16"),"legacy-bf16")
        self.assertEqual(validate_precision_mode("fp32-master"),"fp32-master")
        for x in (None,"fp16","bf16-kahan","FP32",0):
            with self.subTest(x=x),self.assertRaises(ValueError):
                validate_precision_mode(x)

    def test_fp32_cuda_weights_run_with_bf16_autocast(self):
        torch=FakeTorch()
        model=Model("float32")
        self.assertEqual(forward_with_amp(model,1,torch_module=torch),("logits","loss"))
        self.assertEqual(torch.calls,[("cuda","bfloat16",True)])
        self.assertEqual(model.calls,[(1,)])

    def test_legacy_bf16_keeps_existing_path(self):
        torch=FakeTorch()
        model=Model("bfloat16")
        forward_with_amp(model,1,2,torch_module=torch)
        self.assertEqual(torch.calls,[("cuda","bfloat16",False)])
        self.assertEqual(model.calls,[(1,2)])

    def test_fp32_cpu_does_not_enter_cuda_amp(self):
        torch=FakeTorch()
        model=Model("float32",device="cpu")
        forward_with_amp(model,1,torch_module=torch)
        self.assertFalse(torch.calls[0][2])

    def test_selected_fp32_checkpoint_is_bf16_but_resume_keeps_master(self):
        torch=FakeTorch()
        model=Model("float32")
        best=checkpoint_model_state(model,"fp32-master",resume=False,torch_module=torch)
        self.assertEqual(best["weights"].moves,[("to","bfloat16"),("cpu",)])
        model=Model("float32")
        resume=checkpoint_model_state(model,"fp32-master",resume=True,torch_module=torch)
        self.assertEqual(resume["weights"].moves,[("cpu",)])

    def test_legacy_checkpoint_storage_behavior_is_unchanged(self):
        model=Model("bfloat16")
        item=checkpoint_model_state(model,"legacy-bf16",resume=False,torch_module=FakeTorch())
        self.assertEqual(item["weights"].moves,[])

if __name__=="__main__":
    unittest.main()
