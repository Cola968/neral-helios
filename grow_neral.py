"""Function-preserving FFN expansion of our own trained weights only."""
import json
from pathlib import Path
import torch
from model import Neral20M
from architectures import LARGE,expected_parameters
from experiment import load_model,evaluate,read_records,sha,atomic_save

def main():
    torch.set_num_threads(4);torch.manual_seed(40051);device=torch.device('cuda')
    source='export/pilot_02/neral20m.pt';old,p=load_model(source,device);old.eval()
    rows=read_records('curriculum_v2/val.jsonl')
    # Baseline is measured BEFORE constructing/changing the architecture.
    baseline=evaluate(old,rows,device)
    Path('reports/baseline_before_growth.json').write_text(json.dumps({'source_sha256':sha(source),'validation':baseline,'parameters':old.parameter_report(),'data_sha256':sha('curriculum_v2/val.jsonl')},indent=2))
    new=Neral20M(LARGE).to(device);new.eval();original=old.state_dict();expanded=new.state_dict()
    for k,v in original.items():
        if expanded[k].shape==v.shape:expanded[k].copy_(v)
        elif k.endswith(('mlp.gate.weight','mlp.up.weight')):expanded[k][:v.shape[0]].copy_(v)
        elif k.endswith('mlp.down.weight'):
            expanded[k].zero_();expanded[k][:,:v.shape[1]].copy_(v)
        else:raise AssertionError(k)
    new.load_state_dict(expanded,strict=True)
    with torch.inference_mode():
        x=torch.randint(0,260,(2,128),device=device)
        delta=float((old(x)[0]-new(x)[0]).abs().max())
    assert new.parameter_report()['total']==expected_parameters(LARGE)
    assert delta<1e-4,delta
    score=evaluate(new,rows,device)
    print(json.dumps({'fp32_max_delta':delta,'bf16_baseline':baseline,'bf16_expanded':score}),flush=True)
    assert abs(score['macro_loss']-baseline['macro_loss'])<0.01, 'Unexpected BF16 drift' 
    out=Path('seed/neral40m.pt');assert not out.exists(),'Do not overwrite growth seed'
    atomic_save({'model':new.state_dict(),'config':LARGE.to_dict(),'step':0,'format':'neral-growth-v1','source_sha256':sha(source)},out)
    result={'before':baseline,'after_growth_before_training':score,'parameters_before':old.parameter_report(),'parameters_after':new.parameter_report(),'max_abs_logit_difference_fp32':delta,'source_sha256':sha(source),'grown_sha256':sha(out),'method':'FFN 1536 -> 3648, existing parameters copied; added gate/up randomly initialized, added down columns zero. Six layers, 512 width, byte tokenizer unchanged. No foreign model.','seed':40051}
    Path('reports/growth_40m.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
