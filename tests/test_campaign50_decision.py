from campaign50_decision import decide


def _comparison(nis20=40.0, nis50=50.0):
    baseline={
        'utf8_valid_pct':100.0,
        'eos_pct':29.0,
        'repeated_trigram_pct':3.0,
        'rawmath_v1_exact_pct':0.0,
        'v3_pair_overall_pct':55.0,
        'v4_pair_overall_pct':55.0,
        'context_recall_overall_pct':50.0,
        'true_byte_bpb_macro_1024':3.0,
        'v4_math_pct':50.0,
        'v4_code_pct':50.0,
    }
    candidate={
        'utf8_valid_pct':100.0,
        'eos_pct':60.0,
        'repeated_trigram_pct':1.0,
        'rawmath_v1_exact_pct':20.0,
        'v3_pair_overall_pct':60.0,
        'v4_pair_overall_pct':65.0,
        'context_recall_overall_pct':70.0,
        'true_byte_bpb_macro_1024':2.8,
        'v4_math_pct':70.0,
        'v4_code_pct':65.0,
    }
    return {
        'stages':{'baseline20':baseline,'instruct':candidate},
        'nis':{
            'baseline20':{'score':nis20,'status':'complete'},
            'instruct':{'score':nis50,'status':'complete'},
        },
    }


def _v5(single=.5,multi=.5,bpb=3.0):
    return {
        'suite_sha256':'frozen-v5',
        'single_needle':{'accuracy':single},
        'multi_needle':{'accuracy':multi},
        'document_bounded_true_byte_bpb':{
            'macro_bits_per_byte':{'2048':bpb,'3072':bpb,'4095':bpb}
        },
    }


def test_campaign_stops_only_after_hard_gate_nis_and_v5_pass():
    result=decide(_comparison(),_v5(),_v5(.7,.75,2.8))
    assert result['surpassed_20m']
    assert result['promotion']['strict_wins']>=8
    assert result['nis']['strict_win']
    assert result['long_context']['proceed']


def test_campaign_does_not_stop_when_nis_is_not_higher():
    result=decide(_comparison(nis20=50,nis50=50),_v5(),_v5(.7,.75,2.8))
    assert not result['surpassed_20m']
    assert 'nis_not_above_20m' in result['reasons']


def test_campaign_does_not_stop_on_long_context_regression():
    result=decide(_comparison(),_v5(.9,.9,3.0),_v5(.4,.4,3.3))
    assert not result['surpassed_20m']
    assert 'v5_long_context_regression' in result['reasons']
