from frontier350_contract import (
    FRONTIER350_CONFIG,FRONTIER350_PARAMETERS,FRONTIER350_RUNTIME,
    FRONTIER350_TOKENIZER_SHA256,expected_state_shapes,unique_parameter_count,validate_contract,
)

def test_frontier350_contract_is_exact_and_self_consistent():
    shapes=expected_state_shapes()
    assert validate_contract() is True
    assert FRONTIER350_RUNTIME=="frontier350-bpe-v1"
    assert FRONTIER350_PARAMETERS==349_825_152
    assert FRONTIER350_CONFIG["n_layers"]==24
    assert FRONTIER350_CONFIG["vocab_size"]==16_384
    assert len(shapes)==267 and unique_parameter_count()==FRONTIER350_PARAMETERS
    assert shapes["tok_emb.weight"]==shapes["lm_head.weight"]==(16_384,1_152)
    assert len(FRONTIER350_TOKENIZER_SHA256)==64
