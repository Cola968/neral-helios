"""Versioned checkpoint contract and initialization audit for Neral-100M Helix."""
from __future__ import annotations
import hashlib,json,torch
from architectures100_variants import EXPECTED,VARIANTS
from model100 import Neral100M
FORMAT_VERSION=1
TOKENIZER_CONTRACT={"kind":"neral-byte","vocab_size":260,"bos":1,"eos":2,"byte_offset":4}
def stable_hash(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def tokenizer_contract_sha256():return stable_hash(TOKENIZER_CONTRACT)
def init_audit(variant:str,seed:int=100968)->dict:
    if variant not in VARIANTS:raise ValueError('unknown variant')
    torch.manual_seed(seed);m=Neral100M(VARIANTS[variant]);state=m.state_dict();finite=all(torch.isfinite(v).all().item() for v in state.values());tied=m.lm_head.weight.data_ptr()==m.tok_emb.weight.data_ptr();n=sum(p.numel() for p in m.parameters() if p.requires_grad)
    if n!=EXPECTED[variant] or not finite or not tied:raise ValueError('invalid Helix initialization')
    return {"variant":variant,"parameters":n,"finite":finite,"tied_embeddings":tied,"config_sha256":stable_hash(VARIANTS[variant].to_dict()),"tokenizer_contract_sha256":tokenizer_contract_sha256(),"seed":seed}
def checkpoint_payload(model:Neral100M,variant:str,*,tokens:int,step:int,optimizer_state=None,provenance=None)->dict:
    if variant not in VARIANTS or model.cfg!=VARIANTS[variant]:raise ValueError('model/config mismatch')
    if min(tokens,step)<0:raise ValueError('nonnegative counters required')
    o={"format":"neral-helix-checkpoint","format_version":FORMAT_VERSION,"architecture_family":"Neral-100M Helix","variant":variant,"config":model.cfg.to_dict(),"config_sha256":stable_hash(model.cfg.to_dict()),"tokenizer_contract":TOKENIZER_CONTRACT,"tokenizer_contract_sha256":tokenizer_contract_sha256(),"parameters":EXPECTED[variant],"model":model.state_dict(),"tokens":int(tokens),"step":int(step),"provenance":provenance or {},"foreign_weights":False}
    if optimizer_state is not None:o['optimizer']=optimizer_state
    return o
def validate_checkpoint(o:dict,require_optimizer:bool=False)->dict:
    if o.get('format')!='neral-helix-checkpoint' or o.get('format_version')!=FORMAT_VERSION:raise ValueError('unsupported format')
    v=o.get('variant')
    if v not in VARIANTS or o.get('parameters')!=EXPECTED[v]:raise ValueError('parameter contract mismatch')
    if o.get('foreign_weights') is not False:raise ValueError('foreign weights flag')
    if o.get('config_sha256')!=stable_hash(VARIANTS[v].to_dict()) or o.get('tokenizer_contract_sha256')!=tokenizer_contract_sha256():raise ValueError('contract hash mismatch')
    if require_optimizer and 'optimizer' not in o:raise ValueError('optimizer required')
    state=o.get('model')
    if not isinstance(state,dict) or not state or not all(torch.is_tensor(x) and torch.isfinite(x).all().item() for x in state.values()):raise ValueError('invalid state')
    return {"variant":v,"parameters":EXPECTED[v],"valid":True}
def experimental_50m_migration_plan(source_config:dict,target_variant:str)->dict:
    if target_variant not in VARIANTS:raise ValueError('unknown target')
    t=VARIANTS[target_variant]
    return {"status":"experimental_only","production_pretraining":"from_scratch","target_variant":target_variant,"target_parameters":EXPECTED[target_variant],"source_d_model":source_config.get('d_model'),"target_d_model":t.d_model,"allowed_mapping":["overlapping tokenizer rows","exact-shape tensors only"],"forbidden_claim":"function-preserving or equivalent initialization","requires_separate_ablation":True}
