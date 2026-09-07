import collections
import json
from prepare_sft50 import load_rows
from train_instruct50 import encode_sft


def test_builder_uses_exact_trainer_byte_budget(tmp_path):
    p=tmp_path/'source.jsonl'
    prompt='Erkläre kurz eine Zahl.'
    # Multi-byte umlauts catch a character-count approximation of the byte tokenizer.
    rows=[{'id':str(i),'prompt':prompt,'answer':'ä'*i} for i in (10,11,12)]
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    source={'id':'human-de','language':'de','domain':'chat','path':p}
    seq=len(('User: '+prompt+'\nAssistant: '+'ä'*11+'\n').encode('utf8'))+1
    rejected=collections.Counter();clean=load_rows([source],[],rejected,max_seq=seq)
    assert [r['id'] for r in clean]==['10','11']
    assert rejected['pair_exceeds_training_sequence']==1
    for r in clean:
        _,targets=encode_sft(r,seq)
        assert (targets==2).sum()==1
