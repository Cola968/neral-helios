"""Immutable deployment contract for Neral/Arctus Frontier 350M.

App/runtime metadata only. No training, network access or automatic promotion.
"""
from __future__ import annotations

FRONTIER350_RUNTIME="frontier350-bpe-v1"
FRONTIER350_PARAMETERS=349_825_152
FRONTIER350_TOKENIZER_SHA256="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
FRONTIER350_CONFIG={
    "vocab_size":16_384,
    "d_model":1_152,
    "n_layers":24,
    "n_heads":18,
    "n_kv_heads":2,
    "ffn_hidden":3_136,
    "max_seq_len":8_192,
    "rope_theta":250_000.0,
    "rms_eps":1e-5,
    "qk_rms_eps":1e-6,
    "dropout":0.0,
    "initializer_range":0.02,
}

def expected_state_shapes():
    d=FRONTIER350_CONFIG["d_model"];v=FRONTIER350_CONFIG["vocab_size"]
    h=d//FRONTIER350_CONFIG["n_heads"]
    kv=FRONTIER350_CONFIG["n_kv_heads"]*h
    f=FRONTIER350_CONFIG["ffn_hidden"]
    shapes={"tok_emb.weight":(v,d),"norm.weight":(d,),"lm_head.weight":(v,d)}
    for i in range(FRONTIER350_CONFIG["n_layers"]):
        p=f"blocks.{i}."
        shapes.update({
            p+"n1.weight":(d,),p+"attn.q.weight":(d,d),p+"attn.k.weight":(kv,d),
            p+"attn.v.weight":(kv,d),p+"attn.o.weight":(d,d),
            p+"attn.qn.weight":(h,),p+"attn.kn.weight":(h,),
            p+"n2.weight":(d,),p+"ffn.gate.weight":(f,d),
            p+"ffn.up.weight":(f,d),p+"ffn.down.weight":(d,f),
        })
    return shapes

def unique_parameter_count():
    shapes=expected_state_shapes()
    total=0
    for name,shape in shapes.items():
        if name=="lm_head.weight":continue # tied to tok_emb.weight
        n=1
        for size in shape:n*=size
        total+=n
    return total

def validate_contract():
    shapes=expected_state_shapes()
    if len(shapes)!=267:raise RuntimeError("Frontier 350 state-entry contract changed")
    if unique_parameter_count()!=FRONTIER350_PARAMETERS:raise RuntimeError("Frontier 350 parameter contract changed")
    if shapes["tok_emb.weight"]!=shapes["lm_head.weight"]:raise RuntimeError("Frontier 350 tied-head shape changed")
    return True

validate_contract()
