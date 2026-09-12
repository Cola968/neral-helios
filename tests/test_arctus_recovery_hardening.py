import copy
import json
import random
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier350'))
import build_frontier_bpe_data as builder
from train350_compiled import update_token_counts, mask_budget_targets
from recovery_identity import validate_pair
from security_policy350_bpe import _check_shards
from release_security_gate350 import assess, MINIMUMS, MAXIMUMS
from remix_recovery_corpus import slice_stream, source_ranges


@pytest.mark.parametrize('remaining', [1, 2, 7, 8, 9, 31, 32, 33, 10_000_000])
def test_update_budget_exact(remaining):
    counts=update_token_counts(remaining,8,4)
    assert sum(counts)==min(remaining,32)
    assert all(0<x<=8 for x in counts)


def test_partial_accumulation_matches_all_supervised_targets():
    torch.manual_seed(9)
    model=torch.nn.Linear(3,5,bias=False)
    reference=copy.deepcopy(model)
    x=torch.randn(12,3); y=torch.randint(0,5,(12,))
    counts=update_token_counts(9,4,4)
    for i,count in enumerate(counts):
        logits=model(x[i*4:(i+1)*4])
        targets=mask_budget_targets(y[i*4:(i+1)*4].reshape(2,2),count).flatten()
        loss=torch.nn.functional.cross_entropy(logits,targets)
        (loss*count/sum(counts)).backward()
    torch.nn.functional.cross_entropy(reference(x[:9]),y[:9]).backward()
    torch.testing.assert_close(model.weight.grad,reference.weight.grad)


def test_nonfinite_gradient_cannot_mutate_weights():
    p=torch.nn.Parameter(torch.ones(2)); optimizer=torch.optim.AdamW([p])
    p.grad=torch.tensor([float('nan'),0.])
    before=p.detach().clone()
    with pytest.raises(RuntimeError):
        torch.nn.utils.clip_grad_norm_([p],1.,error_if_nonfinite=True)
        optimizer.step()
    torch.testing.assert_close(p,before)
    assert not optimizer.state


@pytest.mark.parametrize('quota', [1,2,3,5,6,7])
def test_source_budget_never_spins_or_invents_eos(tmp_path,monkeypatch,quota):
    monkeypatch.setattr(builder,'stream_rows',lambda *args: iter([{'text':'first'},{'text':'second'}]))
    monkeypatch.setattr(builder,'extract_text',lambda spec,row:row['text'])
    monkeypatch.setattr(builder,'quality_ok',lambda text:(True,'ok'))
    class Tok:
        special_ids={'<|eos|>':1}
        def encode_document(self,text):return [0,4,5,6,1]
    from build_frontier_data import DEFAULT_SOURCES
    seen=builder.PersistentSeen(tmp_path/'seen.db')
    writer=builder.StreamingShardWriter(tmp_path/'train','train',20)
    try:report=builder.build_source(DEFAULT_SOURCES[0],'a'*40,quota,writer,seen,Tok())
    finally:seen.close()
    path=tmp_path/'train'/writer.finish()[0]['path']
    assert np.fromfile(path,dtype='<u2').tolist()==([0,4,5,6,1]*2)[:quota]
    assert report['synthetic_eos_targets']==0


def identities():
    def report(sha,protocol):return {'architecture':'frontier350-v1','checkpoint_sha256':sha*64,
        'tokenizer_sha256':'a'*64,'protocol_sha256':protocol*64,'checkpoint_processed_tokens':1}
    return [report('b','d'),report('b','e'),report('c','d'),report('c','e')]


def test_matching_evaluation_identity():validate_pair(*identities())


@pytest.mark.parametrize('field,value',[('checkpoint_sha256','d'*64),('protocol_sha256','f'*64),('tokenizer_sha256','b'*64),('checkpoint_processed_tokens',2)])
def test_mixed_evidence_rejected(field,value):
    reports=identities();reports[-1][field]=value
    with pytest.raises(ValueError):validate_pair(*reports)


def test_legacy_unidentified_evidence_rejected():
    reports=identities();del reports[0]['checkpoint_sha256']
    with pytest.raises(ValueError):validate_pair(*reports)


@pytest.mark.parametrize('extra,duplicate',[(True,False),(False,True)])
def test_unlisted_and_duplicate_shards_rejected(tmp_path,extra,duplicate):
    root=tmp_path/'train';root.mkdir();p=root/'a.bin';p.write_bytes(b'\x04\x00'*8)
    item={'path':p.name,'tokens':8,'sha256':builder.sha256(p)}
    if extra:(root/'unlisted.bin').write_bytes(p.read_bytes())
    with pytest.raises(ValueError):_check_shards(tmp_path,'train',[item,item] if duplicate else [item])


@pytest.mark.parametrize('value',[-1,101,float('nan'),float('inf'),True])
def test_security_gate_rejects_invalid_percentage(value):
    scores=dict(MINIMUMS,**MAXIMUMS);scores['canary_extraction_rate']=value
    assert not assess(scores)['pass']


def test_remix_crosses_shards_without_changing_tokens():
    arrays=[np.arange(7),np.arange(7,17),np.arange(17,30)]
    assert np.concatenate(list(slice_stream(arrays,5,15))).tolist()==list(range(5,20))


def test_validation_reports_never_become_train_ranges():
    reports=[{'partition':'valid','source':{'name':'held_out'},'emitted_tokens':8},
             {'partition':'train','source':{'name':'edu'},'emitted_tokens':12}]
    assert set(source_ranges({'sources':reports,'train_tokens':12}))=={'edu'}


@pytest.mark.parametrize('regression',[False,True])
def test_selector_no_category_compensation(tmp_path,monkeypatch,regression):
    import select_recovery_lr_pilot as selector
    baseline,val,candidate,candidate_val=identities()
    cats=['bilingual','code','factual','language','math','reasoning','science']
    for dev in (baseline,candidate):
        dev.update(suite='arctus-recovery-dev-v1',cases=140,overall_accuracy_pct=50.,
                   categories={k:{'correct':10,'total':20,'accuracy_pct':50.} for k in cats})
    val['validation_loss']=3.3;candidate_val['validation_loss']=3.2
    if regression:
        candidate['categories']['bilingual'].update(correct=9,accuracy_pct=45.)
        candidate['categories']['math'].update(correct=11,accuracy_pct=55.)
    paths=[]
    for i,obj in enumerate((baseline,val,candidate,candidate_val)):
        p=tmp_path/f'{i}.json';p.write_text(json.dumps(obj));paths.append(str(p))
    out=tmp_path/'selection.json'
    monkeypatch.setattr(sys,'argv',['selector','--baseline-dev',paths[0],'--baseline-validation',paths[1],
                                  '--candidate','control','0.000075',paths[2],paths[3],'--out',str(out)])
    selector.main()
    result=json.loads(out.read_text())
    assert result['selection_allowed'] is (not regression)


def test_inconsistent_scores_rejected():
    from select_recovery_lr_pilot import verify_counts
    with pytest.raises(ValueError):
        verify_counts({'categories':{'math':{'total':20,'correct':1,'accuracy_pct':100.}},'cases':20,'overall_accuracy_pct':100.})
