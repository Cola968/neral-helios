import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier350'))
import pytest
import torch
from protection_v9 import project_descent,bound_step,configure_trainable
from model350 import RMSNorm,Neral350M,Neral350Config


def test_conflicting_direction_projected():
    refs=[torch.tensor([1.,0.]),torch.tensor([0.,1.])]
    result,report=project_descent(torch.tensor([-2.,3.]),refs)
    torch.testing.assert_close(result,torch.tensor([0.,3.]))
    assert all(torch.dot(result,r)>=0 for r in refs)


def test_infeasible_bounded_iteration_returns_safe_zero():
    result,report=project_descent(torch.tensor([-2.,-3.]),[torch.tensor([1.,0.])],max_passes=0)
    assert report['zero_fallback'] and not result.any()


def test_zero_reference_and_nonfinite():
    vec=torch.tensor([1.,2.])
    result,_=project_descent(vec,[torch.zeros(2)])
    torch.testing.assert_close(result,vec)
    with pytest.raises(FloatingPointError):project_descent(torch.tensor([float('nan')]),[])


def test_step_bound_preserves_halfspaces():
    result,scale=bound_step(torch.tensor([3.,4.]),torch.tensor([3.,4.]),.1)
    assert scale==pytest.approx(.1)
    assert result.norm()<=.500001


def test_bf16_norm_identical_and_fp32_master_keeps_activation_dtype():
    torch.manual_seed(12);x=torch.randn(2,3,8).bfloat16();m=RMSNorm(8,1e-5).bfloat16()
    expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+m.eps)).to(x.dtype)*m.weight
    torch.testing.assert_close(m(x),expected,rtol=0,atol=0)
    m.float();y=m(x);assert y.dtype==torch.bfloat16
    y.float().sum().backward();assert m.weight.grad.dtype==torch.float32 and torch.isfinite(m.weight.grad).all()


def test_only_requested_top_blocks_change():
    cfg=Neral350Config(vocab_size=32,d_model=32,n_layers=4,n_heads=4,n_kv_heads=2,ffn_hidden=64,max_seq_len=32)
    model=Neral350M(cfg).float();original=copy.deepcopy(model.state_dict())
    names=configure_trainable(model,2)
    assert all(n.startswith(('blocks.2.','blocks.3.')) for n,p in names)
    opt=torch.optim.AdamW([p for _,p in names],lr=.01)
    _,loss=model(torch.randint(0,32,(2,12)),torch.randint(0,32,(2,12)))
    loss.backward();opt.step()
    trainable={n for n,_ in names}
    assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items() if k not in trainable)
    assert any(not torch.equal(model.state_dict()[k],original[k]) for k in trainable)


def test_interpolation_cannot_restore_unaligned_masters(tmp_path):
    from interpolate_recovery_checkpoint import interpolate
    from model350 import FRONTIER_350
    anchor={'config':FRONTIER_350.to_dict(),'model':{'w':torch.tensor([1.],dtype=torch.bfloat16)},
            'tokenizer_sha256':'a'*64,'processed_tokens':1,'optimizer':{'state':{},'param_groups':[]}}
    candidate=copy.deepcopy(anchor);candidate['model']['w'].fill_(2.);candidate['processed_tokens']=2
    candidate['v9_training_state']={'masters':{'w':torch.tensor([2.])}}
    a,b,c=(tmp_path/name for name in ('anchor.pt','candidate.pt','aligned.pt'))
    torch.save(anchor,a);torch.save(candidate,b);interpolate(a,b,c,.4)
    result=torch.load(c,weights_only=False)
    assert 'v9_training_state' not in result
    torch.testing.assert_close(result['model']['w'],torch.tensor([1.4],dtype=torch.bfloat16))


def test_bilingual_gate_rejects_missing_and_mixed_evidence():
    from run_guarded_v9 import strict_bilingual
    def report(sha):return {'cases':400,'directions':{d:{'total':200,'correct':100,'accuracy_pct':50.} for d in ('de_to_en','en_to_de')},
        'overall_accuracy_pct':50.,'cases_sha256':'a'*64,'tokenizer_sha256':'b'*64,'protocol_sha256':'c'*64,
        'checkpoint_sha256':sha,'checkpoint_processed_tokens':2}
    baseline=report('d'*64);candidate=report('e'*64)
    dev={k:candidate[k] for k in ('tokenizer_sha256','checkpoint_sha256','checkpoint_processed_tokens')}
    assert strict_bilingual(baseline,candidate,dev)
    dev['checkpoint_sha256']='f'*64
    with pytest.raises(ValueError):strict_bilingual(baseline,candidate,dev)
    dev['checkpoint_sha256']=candidate['checkpoint_sha256'];candidate['overall_accuracy_pct']=float('nan')
    with pytest.raises(ValueError):strict_bilingual(baseline,candidate,dev)
