import math

from pretrain50_stage_b import (
    build_schedule,
    build_validation_schedule,
    cosine_lr,
    document_blocks,
)


def synthetic_documents():
    # Offsets are relative to each split/language stream, matching prepare_pretrain50.py.
    return [
        {'split':'train','language':'de','start':0,'end':1026},
        {'split':'train','language':'de','start':1026,'end':2052},
        {'split':'train','language':'en','start':0,'end':1026},
        {'split':'val','language':'de','start':0,'end':1026},
        {'split':'val','language':'de','start':1026,'end':2052},
        {'split':'val','language':'en','start':0,'end':1026},
        {'split':'test','language':'de','start':0,'end':1026},
    ]


def test_document_blocks_never_cross_boundaries_and_cover_tail():
    rows=synthetic_documents();seq=256
    blocks=document_blocks(rows,'de',seq,'train')
    assert blocks
    de_docs=[r for r in rows if r['split']=='train' and r['language']=='de']
    for _,start in blocks:
        assert any(r['start']<=start and start+seq<r['end'] for r in de_docs)
    for r in de_docs:
        assert ('de',r['end']-seq-1) in blocks


def test_document_blocks_respect_requested_split():
    rows=synthetic_documents();seq=256
    train=set(document_blocks(rows,'de',seq,'train'))
    val=set(document_blocks(rows,'de',seq,'val'))
    # Offsets can numerically overlap because every split has its own memmap, so verify
    # membership against the correct split documents rather than comparing offsets alone.
    for _,start in val:
        assert any(r['split']=='val' and r['language']=='de' and r['start']<=start and start+seq<r['end'] for r in rows)
    assert train and val


def test_schedule_is_deterministic_and_balanced():
    rows=synthetic_documents();seq=256
    a,ma=build_schedule(rows,seq,.60,123)
    b,mb=build_schedule(rows,seq,.60,123)
    assert a==b and ma==mb
    assert ma['de_unique_blocks']>0 and ma['en_unique_blocks']>0
    assert ma['en_scheduled_blocks']>=ma['en_unique_blocks']
    assert ma['english_replay_factor']>=1.0
    assert abs(ma['actual_de_fraction']-.60)<.08
    assert not ma['document_boundary_crossing'] and ma['padding_tokens']==0


def test_validation_schedule_is_deterministic_balanced_and_val_only():
    rows=synthetic_documents();seq=256
    a,ma=build_validation_schedule(rows,seq,3,999)
    b,mb=build_validation_schedule(rows,seq,3,999)
    assert a==b and ma==mb
    assert set(a)=={'de','en'}
    assert len(a['de'])==3 and len(a['en'])==3
    assert ma['split']=='val' and ma['document_boundary_crossing'] is False
    for lang,items in a.items():
        docs=[r for r in rows if r['split']=='val' and r['language']==lang]
        for _,start in items:
            assert any(r['start']<=start and start+seq<r['end'] for r in docs)


def test_cosine_lr_is_bounded_and_decreasing():
    peak=2e-4;floor=2e-5
    values=[cosine_lr(p,peak,floor) for p in [0,.25,.5,.75,1]]
    assert math.isclose(values[0],peak)
    assert math.isclose(values[-1],floor)
    assert all(a>=b for a,b in zip(values,values[1:]))
    assert all(floor<=v<=peak for v in values)
