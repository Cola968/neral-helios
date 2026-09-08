from architectures100_variants import EXPECTED, HELIX_A2, report
from architectures100 import expected_parameters100


def test_a2_exact_parameter_contract():
    assert expected_parameters100(HELIX_A2) == EXPECTED['a2'] == 100_470_912


def test_a2_compact_cache_contract():
    r=report('a2')
    assert r['kv_cache_bf16_4096_bytes'] == 48*1024*1024
    assert r['kv_cache_bf16_8192_bytes'] == 96*1024*1024
