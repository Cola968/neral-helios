"""CPU validation and frozen-holdout separation for QLoRA-v2 pilot."""
import json
import pytest
from pathlib import Path
from scripts.neral_qlora_v2_pilot import load_splits, training_target, to_messages, SYSTEM
from scripts.neral_qlora_v2_compare40 import judge, frozen_inputs


ROOT=Path(__file__).resolve().parents[1]/"datasets"/"neral_cpu_foundation"
def test_exact_184_train_and_32_val_without_opening_test(monkeypatch):
    original=Path.read_bytes
    def guard(self,*a,**k):
        if self.name.endswith(".test.jsonl"):
            raise AssertionError("Test split must be untouched during training")
        return original(self,*a,**k)
    monkeypatch.setattr(Path,"read_bytes",guard)
    train,val=load_splits(ROOT/"helios_terra_v2.jsonl",
                          ROOT/"helios_terra_v2.val.jsonl")
    assert (len(train),len(val))==(184,32)
    assert not ({r["id"] for r in train}&{r["id"] for r in val})


def test_exact_tool_json_target_and_native_messages():
    train,val=load_splits(ROOT/"helios_terra_v2.jsonl",
                          ROOT/"helios_terra_v2.val.jsonl")
    messages=to_messages(train[:1])[0]["messages"]
    assert len(messages)==3 and messages[0]["content"]==SYSTEM
    actual=json.loads(messages[2]["content"])
    assert set(actual)=={"tool","spec","answer"}
    assert actual=={"tool":train[0]["task"]["tool"],
                    "spec":train[0]["task"]["spec"],"answer":train[0]["answer"]}


def test_true_tool_match_and_invalid_format_fails():
    train,_=load_splits(ROOT/"helios_terra_v2.jsonl",
                         ROOT/"helios_terra_v2.val.jsonl")
    row=train[0]
    assert judge(row,training_target(row))==(True,False)
    assert judge(row,"not JSON")==(False,True)
    wrong=json.loads(training_target(row))
    wrong["answer"]="UNTRUE"
    assert judge(row,json.dumps(wrong))==(False,False)


def test_test_split_unreadable_before_adapter_frozen(tmp_path,monkeypatch):
    marker=tmp_path/"test.jsonl"
    original=Path.read_bytes
    def guard(self,*a,**k):
        if self==marker:raise AssertionError("Test was opened before frozen adapter")
        return original(self,*a,**k)
    monkeypatch.setattr(Path,"read_bytes",guard)
    with pytest.raises(FileNotFoundError):
        frozen_inputs(trained=tmp_path/"no-trained-adapter",
                      v1=tmp_path/"no-v1",test=marker,
                      train=ROOT/"helios_terra_v2.jsonl",
                      val=ROOT/"helios_terra_v2.val.jsonl",
                      output=tmp_path/"report.json")


def test_line_ending_normalization_does_not_mask_modified_dataset(tmp_path):
    import hashlib
    from scripts.neral_qlora_v2_pilot import _digest
    f=tmp_path/"tiny.jsonl"
    raw=b'{"i":1}\n{"i":2}\n'
    expected=hashlib.sha256(raw).hexdigest()
    f.write_bytes(raw.replace(b"\n",b"\r\n"))
    assert _digest(f)==expected
    f.write_bytes(raw.replace(b'2',b'3').replace(b"\n",b"\r\n"))
    assert _digest(f)!=expected
    f.write_bytes(raw.replace(b"\n",b"\r"))
    with pytest.raises(ValueError):
        _digest(f)
