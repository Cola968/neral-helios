import hashlib,pickle,sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import import_frontier350 as imp

class FakeFinite:
    def all(self):return True

class FakeTensor:
    def __init__(self,shape,dtype="bf16"):self.shape=shape;self.dtype=dtype

def fake_state():
    return {name:FakeTensor(shape) for name,shape in imp.expected_state_shapes().items()}

class FakeTorch:
    bfloat16="bf16";float32="fp32"
    def __init__(self,payload,fail_safe=False):
        self.payload=payload;self.fail_safe=fail_safe;self.calls=[]
    def load(self,*args,**kwargs):
        self.calls.append(kwargs.get("weights_only"))
        if kwargs.get("weights_only") and self.fail_safe:
            raise pickle.UnpicklingError("legacy fixture")
        return self.payload
    def is_tensor(self,value):return isinstance(value,FakeTensor)
    def isfinite(self,value):return FakeFinite()
    def equal(self,a,b):return tuple(a.shape)==tuple(b.shape)
    def save(self,payload,path):
        self.saved=payload
        Path(path).write_bytes(b"sanitized-deployment")

def test_unsafe_pickle_fallback_refused_for_unknown_sha(tmp_path,monkeypatch):
    source=tmp_path/"source.pt";source.write_bytes(b"source")
    tokenizer=tmp_path/"tokenizer.json";tokenizer.write_bytes(b"tokenizer")
    state=fake_state()
    payload={"config":imp.FRONTIER350_CONFIG,"tokenizer_sha256":imp.FRONTIER350_TOKENIZER_SHA256,
             "processed_tokens":123,"model":state}
    torch=FakeTorch(payload,fail_safe=True)
    monkeypatch.setitem(sys.modules,"torch",torch)
    actual=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(imp,"sha256_file",lambda p: actual if Path(p)==source else imp.FRONTIER350_TOKENIZER_SHA256)
    with pytest.raises(ValueError,match="unsupported unsafe deserialization"):
        imp.validate_frontier_checkpoint(source,actual,tokenizer,imp.FRONTIER350_TOKENIZER_SHA256)
    assert torch.calls==[True]

def test_known_historical_sha_may_fallback_then_reports_it(tmp_path,monkeypatch):
    source=tmp_path/"source.pt";source.write_bytes(b"source")
    tokenizer=tmp_path/"tokenizer.json";tokenizer.write_bytes(b"tokenizer")
    state=fake_state()
    payload={"config":imp.FRONTIER350_CONFIG,"tokenizer_sha256":imp.FRONTIER350_TOKENIZER_SHA256,
             "processed_tokens":2_073_161_216,"model":state}
    torch=FakeTorch(payload,fail_safe=True)
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setattr(imp,"sha256_file",lambda p:
        imp.TRUSTED_HISTORICAL_CONTINUITY_SHA if Path(p)==source else imp.FRONTIER350_TOKENIZER_SHA256)
    verified,returned=imp.validate_frontier_checkpoint(
        source,imp.TRUSTED_HISTORICAL_CONTINUITY_SHA,tokenizer,imp.FRONTIER350_TOKENIZER_SHA256)
    assert returned is state
    assert torch.calls==[True,False]
    assert verified["historical_pickle_fallback"] is True
    assert verified["processed_tokens"]==2_073_161_216

def test_import_writes_sanitized_inference_payload_not_source_copy(tmp_path,monkeypatch):
    source=tmp_path/"historical.pt";source.write_bytes(b"historical-training-pickle")
    tokenizer=tmp_path/"tokenizer.json";tokenizer.write_bytes(b"own-tokenizer")
    state={"weight":"fixture"}
    verified={"source_checkpoint_sha256":"a"*64,"tokenizer_sha256":imp.FRONTIER350_TOKENIZER_SHA256,
              "processed_tokens":2_073_161_216,"dtype":"bfloat16","state_entries":267,
              "parameters":imp.FRONTIER350_PARAMETERS,"historical_pickle_fallback":True}
    monkeypatch.setattr(imp,"validate_frontier_checkpoint",lambda *a,**k:(verified,state))
    torch=FakeTorch({})
    monkeypatch.setitem(sys.modules,"torch",torch)
    original_sha=imp.sha256_file
    monkeypatch.setattr(imp,"sha256_file",lambda p:
        imp.FRONTIER350_TOKENIZER_SHA256 if Path(p).name=="tokenizer.json" else original_sha(Path(p)))
    report=imp.import_frontier350(
        source,"a"*64,tokenizer,imp.FRONTIER350_TOKENIZER_SHA256,
        "neral-arctus-350","Arctus 350M",tmp_path/"assets")
    model=tmp_path/"assets/installed_models/neral-arctus-350/model.pt"
    assert model.read_bytes()==b"sanitized-deployment"
    assert model.read_bytes()!=source.read_bytes()
    assert torch.saved["model"] is state
    assert set(torch.saved)=={"model","config","tokenizer_sha256","processed_tokens","source_sha256","deployment_format"}
    assert torch.saved["deployment_format"]=="frontier350-inference-v1"
    assert report["sanitized_inference_export"] is True
    assert report["deployment_sha256"]==hashlib.sha256(b"sanitized-deployment").hexdigest()
