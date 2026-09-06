"""Only the two explicitly supported own Neral configurations."""
from config import NeralConfig
SMALL=NeralConfig()
LARGE=NeralConfig(ffn_hidden=3648)
FIFTY=NeralConfig(d_model=624,n_layers=8,n_heads=12,ffn_hidden=2496)
def expected_parameters(cfg):
    if cfg.to_dict()==SMALL.to_dict():return 20_587_008
    if cfg.to_dict()==LARGE.to_dict():return 40_051_200
    if cfg.to_dict()==FIFTY.to_dict():return 50_012_976
    raise ValueError('Unbekannte Neral-Architektur.')
