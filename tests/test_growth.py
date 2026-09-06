from pathlib import Path
import pytest,torch
from architectures import LARGE,expected_parameters
from config import NeralConfig
from experiment import load_model,sha

def test_only_two_own_architectures_allowed():
    assert expected_parameters(LARGE)==40_051_200
    assert expected_parameters(NeralConfig())==20_587_008
    with pytest.raises(ValueError):expected_parameters(NeralConfig(n_layers=12))

def test_40m_exact_count_and_inherited_weights_unchanged():
    m,p=load_model('export/neral40m/neral40m.pt',torch.device('cpu'))
    assert m.parameter_report()['trainable']==40_051_200
    small=torch.load('export/pilot_02/neral20m.pt',map_location='cpu',weights_only=True)['model']
    for key,v in small.items():
        own=p['model'][key]
        if own.shape!=v.shape:own=own[:,:1536] if key.endswith('down.weight') else own[:1536]
        assert torch.equal(own,v),key
    assert p['validation_selected'] is False
    assert sha('export/neral40m/neral40m.pt')==Path('export/neral40m/neral40m.sha256').read_text().split()[0]
    assert torch.count_nonzero(p['model']['blocks.0.mlp.down.weight'][:,1536:])>0
