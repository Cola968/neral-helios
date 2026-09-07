from long_context_gate50 import decide


def _report(single=.5,multi=.5,bpb2048=2.0,bpb3072=2.1,bpb4095=2.2,sha='a'*64):
    return {
        'suite_sha256':sha,
        'single_needle':{'accuracy':single},
        'multi_needle':{'accuracy':multi},
        'document_bounded_true_byte_bpb':{'macro_bits_per_byte':{'2048':bpb2048,'3072':bpb3072,'4095':bpb4095}},
    }


def test_protect_gate_allows_small_tradeoffs():
    before=_report();after=_report(single=.45,multi=.5,bpb2048=2.01,bpb3072=2.1,bpb4095=2.2)
    result=decide(before,after,'protect')
    assert result['proceed']
    assert not result['severe']


def test_gate_rejects_large_long_bpb_regression():
    before=_report();after=_report(bpb4095=2.4)
    result=decide(before,after,'protect')
    assert not result['proceed']
    assert 'bpb_4095_regressed_gt_5pct' in result['severe']


def test_progress_gate_requires_actual_long_context_gain():
    before=_report();same=_report()
    assert not decide(before,same,'progress')['proceed']
    better=_report(single=.6,bpb4095=2.1)
    assert decide(before,better,'progress')['proceed']


def test_gate_fails_on_suite_mismatch():
    result=decide(_report(sha='a'*64),_report(sha='b'*64),'protect')
    assert not result['proceed']
    assert result['reason']=='suite_mismatch'


def test_gate_fails_closed_when_long_bpb_is_missing():
    before=_report();after=_report()
    del before['document_bounded_true_byte_bpb']['macro_bits_per_byte']['3072']
    result=decide(before,after,'protect')
    assert not result['proceed']
    assert result['reason']=='missing_metrics'
    assert 'before_bpb_3072' in result['missing']
