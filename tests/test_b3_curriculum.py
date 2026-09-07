import json

from stage_progress_gate50 import decide
from prepare_pretrain50_b3 import load_b2_exclusions, select_disjoint_phases


def test_b3_phase_selection_is_document_disjoint():
    sources=[{'id':'a','target_fraction':.5},{'id':'b','target_fraction':.5}]
    rows=[]
    for sid in ('a','b'):
        for i in range(20):
            rows.append({'source_id':sid,'id':str(i),'base_split':'train','tokens':100,'selection_hash':f'{i:064x}'})
    selected,meta=select_disjoint_phases(rows,sources,1000,1000)
    a={(r['source_id'],r['id']) for r in selected['a']};b={(r['source_id'],r['id']) for r in selected['b']}
    assert a.isdisjoint(b)
    assert sum(r['tokens'] for r in selected['a'])>=1000
    assert sum(r['tokens'] for r in selected['b'])>=1000
    assert meta['a']['a']['selected_tokens']>=500


def test_b3_loads_hash_row_and_url_exclusions_from_b2(tmp_path):
    digest='a'*64
    rows=[
        {'source_id':'fineweb-en-general','id':'doc-1','normalized_sha256':digest,'url':'https://example.test/doc/'},
        {'source_id':'fineweb-en-general','id':'doc-2','normalized_sha256':'b'*64,'url':None},
    ]
    (tmp_path/'documents.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    excluded=load_b2_exclusions(tmp_path)
    assert digest in excluded['hashes']
    assert ('fineweb-en-general','doc-1') in excluded['source_rows']
    assert 'https://example.test/doc' in excluded['urls']


def test_progress_gate_rejects_repetition_gain_bought_with_bpb_loss():
    before={'eos_pct':20,'rawmath_v1_exact_pct':10,'v3_pair_overall_pct':55,'v4_pair_overall_pct':55,'context_recall_overall_pct':40,'repeated_trigram_pct':20,'true_byte_bpb_macro_1024':3.0}
    after=before|{'eos_pct':30,'rawmath_v1_exact_pct':20,'repeated_trigram_pct':5,'true_byte_bpb_macro_1024':3.06}
    result=decide({'before':before,'after':after})
    assert not result['proceed']
    assert 'repetition_improved_but_bpb_worsened_gt_1pct' in result['severe']


def test_progress_gate_accepts_broad_nonregressive_progress():
    before={'eos_pct':20,'rawmath_v1_exact_pct':10,'v3_pair_overall_pct':55,'v4_pair_overall_pct':55,'context_recall_overall_pct':40,'repeated_trigram_pct':20,'true_byte_bpb_macro_1024':3.0}
    after={'eos_pct':25,'rawmath_v1_exact_pct':12,'v3_pair_overall_pct':57,'v4_pair_overall_pct':56,'context_recall_overall_pct':45,'repeated_trigram_pct':15,'true_byte_bpb_macro_1024':2.9}
    result=decide({'before':before,'after':after})
    assert result['proceed']
    assert result['wins']>=3
