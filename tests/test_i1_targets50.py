from copy import deepcopy
from i1_targets50 import evaluate


def inputs():
    metrics={
        'utf8_valid_pct':99,
        'rawmath_v1_exact_pct':60,
        'rawmath_transfer_ood_pct':45,
        'eos_pct':60,
        'repeated_trigram_pct':4.99,
        'v3_code_pct':60,
        'v4_code_pct':60,
        'v4_math_pct':50,
        'v3_de_grammar_pct':60,
        'v4_de_grammar_pct':60,
        'v3_pair_overall_pct':60,
        'v4_pair_overall_pct':60,
        'context_recall_overall_pct':60,
    }
    comp={'stages':{'baseline20':metrics,'instruct':dict(metrics)},
          'nis':{'baseline20':{'score':44},'instruct':{'score':55}}}
    v5={'suite_sha256':'fixed','single_needle':{'accuracy':.6},'multi_needle':{'accuracy':.6},
        'document_bounded_true_byte_bpb':{'macro_bits_per_byte':{str(n):3. for n in (2048,3072,4095)}}}
    return comp,v5,deepcopy(v5),{'ready_for_rc_review':True}


def test_exact_thresholds_and_strict_repetition():
    args=inputs();assert evaluate(*args)['targets_met']
    args[0]['stages']['instruct']['repeated_trigram_pct']=5
    assert not evaluate(*args)['targets_met']


def test_missing_nonfinite_or_regressed_context_never_passes():
    args=inputs();args[2]['multi_needle']['accuracy']=.59
    assert not evaluate(*args)['targets_met']
    for invalid in (None,float('nan'),float('inf'),True):
        args=inputs();args[0]['stages']['instruct']['eos_pct']=invalid
        assert not evaluate(*args)['targets_met']


def test_strengthened_math_code_and_transfer_targets_are_mandatory():
    for key,value in (
        ('rawmath_transfer_ood_pct',44.99),
        ('v3_code_pct',49.99),
        ('v4_code_pct',49.99),
        ('v4_math_pct',49.99),
        ('utf8_valid_pct',98.99),
    ):
        args=inputs();args[0]['stages']['instruct'][key]=value
        assert not evaluate(*args)['targets_met']


def test_nis_alone_cannot_override_promotion():
    args=inputs();args[3]['ready_for_rc_review']=False
    assert not evaluate(*args)['targets_met']
