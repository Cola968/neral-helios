from rawmath_progress_gate50 import decide as rawmath_decide


def base():
    return {
        'rawmath_v1_exact_pct':10.0,
        'rawmath_distribution_id_pct':12.0,
        'rawmath_distribution_ood_pct':4.0,
        'rawmath_distribution_transfer_pct':3.0,
        'true_byte_bpb_macro_1024':3.0,
        'repeated_trigram_pct':8.0,
        'v3_pair_overall_pct':55.0,
        'v4_pair_overall_pct':55.0,
    }


def test_rawmath_gate_requires_broad_math_gain_without_language_regression():
    before=base();after=before|{
        'rawmath_v1_exact_pct':25.0,
        'rawmath_distribution_id_pct':28.0,
        'rawmath_distribution_ood_pct':12.0,
        'rawmath_distribution_transfer_pct':8.0,
        'true_byte_bpb_macro_1024':2.98,
    }
    r=rawmath_decide({'before':before,'after':after})
    assert r['proceed'] and r['math_slice_wins']>=2


def test_rawmath_gate_rejects_math_gain_with_bpb_regression():
    before=base();after=before|{
        'rawmath_v1_exact_pct':25.0,
        'rawmath_distribution_id_pct':28.0,
        'rawmath_distribution_ood_pct':12.0,
        'rawmath_distribution_transfer_pct':8.0,
        'true_byte_bpb_macro_1024':3.2,
    }
    r=rawmath_decide({'before':before,'after':after})
    assert not r['proceed']
    assert 'bpb_regressed_gt_3pct' in r['severe']


def test_rawmath_gate_fails_closed_on_missing_ood():
    before=base();after=base();del after['rawmath_distribution_ood_pct']
    r=rawmath_decide({'before':before,'after':after})
    assert not r['proceed'] and 'rawmath_distribution_ood_pct' in r['missing']
