"""Tokenize staged SFT candidates ONLY for capacity/mask preflight.

No model weights are loaded, gradients computed, pack generated, text logged,
or approval granted. Uses the SHA-pinned native tokenizer and original SFT encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sft_contract import encode_row
from audit_sft_stage import load

TOKENIZER_SHA='4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd'
TRAIN_SHA='58a4ef28db625b928a931f560978576eb216a603710b8ecb306e6c8f0aeb6f4e'
VALID_SHA='e34bd1cb9bde72048535769f3ad49516b5fa7d0d868db27fb7068837862fac26'


def sha(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def run(train:Path,valid:Path,tokenizer_path:Path):
    for path,expected in ((train,TRAIN_SHA),(valid,VALID_SHA),(tokenizer_path,TOKENIZER_SHA)):
        if sha(path)!=expected:
            raise ValueError('input file does not match pinned independent SHA')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    bos=tokenizer.token_to_id('<|bos|>')
    eos=tokenizer.token_to_id('<|eos|>')
    if tokenizer.get_vocab_size()!=16384 or bos is None or eos is None or bos==eos:
        raise ValueError('incompatible own tokenizer')
    report={'schema':1,'classification':'TOKENIZATION_READ_ONLY_NOT_APPROVED_NOT_TRAINED',
            'tokenizer_sha256':TOKENIZER_SHA,'candidate_train_sha256':TRAIN_SHA,
            'candidate_validation_sha256':VALID_SHA,'context':512,
            'splits':{},'invalid_ids_no_text':[],
            'approved_for_training':False,'training_started':False}
    for split,path in (('train',train),('validation',valid)):
        rows=load(path)
        length=[]; answers=[]
        for row in rows:
            try:
                pack=encode_row(row,tokenizer,int(bos),int(eos),16384,512)
                length.append(len(pack['input_ids']))
                answers.append(pack['answer_tokens_including_eos'])
                if pack['labels'][-1]!=eos or pack['labels'].count(eos)!=1:
                    raise ValueError('EOS target alignment failure')
            except (ValueError,AssertionError) as exc:
                report['invalid_ids_no_text'].append({'id':row['id'],'split':split,'error_type':type(exc).__name__})
        report['splits'][split]={'source_rows':len(rows),'valid_512_rows':len(length),
                                  'max_sequence_tokens':max(length) if length else None,
                                  'max_answer_tokens_including_eos':max(answers) if answers else None,
                                  'supervised_answer_tokens_including_eos':sum(answers)}
    if not report['splits']['train']['valid_512_rows'] or not report['splits']['validation']['valid_512_rows']:
        raise ValueError('insufficient tokenizable data')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train','validation','tokenizer','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('refusing to overwrite tokenization preflight')
    report=run(a.train,a.validation,a.tokenizer)
    with a.out.open('x',encoding='utf8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='invalid_ids_no_text'},ensure_ascii=False),flush=True)
    print('invalid_count='+str(len(report['invalid_ids_no_text'])),flush=True)

if __name__=='__main__':main()
