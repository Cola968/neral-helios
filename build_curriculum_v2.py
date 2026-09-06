"""Small, provenance-preserving human-dialogue extension; no model downloads."""
import collections,gzip,hashlib,json,re,unicodedata
from pathlib import Path
from experiment import encode_record,read_records,sha

SOURCE=Path('corpus_sources/oasst1')
OUT=Path('curriculum_v2')
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def main():
    OUT.mkdir(exist_ok=True)
    records={s:read_records(Path('curriculum')/(s+'.jsonl')) for s in ['train','val','test']}
    seen={norm(r['prompt']) for rs in records.values() for r in rs}
    answers={norm(r['answer']) for rs in records.values() for r in rs}
    rejected=collections.Counter();kept=[]
    # Issues spotted while inspecting source samples; source ratings are not factual verification.
    exclusions=['persien','aktienkurs','multivitaminsaft','schwarzen tasten','usb-ladekabel','blaubarsch','softdrinks','stable diffusion','aktuellen zeitpunkt','youtube','gleise','hopr']
    for line in gzip.open(SOURCE/'2023-04-12_oasst_ready.trees.jsonl.gz','rt',encoding='utf-8'):
        tree=json.loads(line);q=tree['prompt']
        if q['lang'] not in ['de','en']:continue
        for a in q.get('replies',[]):
            if a.get('rank')!=0 or a.get('lang')!=q['lang']:continue
            reason=None
            if any(n.get('synthetic') is not False or n.get('deleted') or n.get('review_result') is not True or n.get('model_name') for n in [q,a]):reason='not_approved_human'
            elif any(n.get('labels',{}).get(k,{}).get('value',1)>0 for n in [q,a] for k in ['spam','pii','hate_speech','sexual_content','lang_mismatch']):reason='source_flags_or_missing_labels'
            else:
                labels=a.get('labels',{})
                if labels.get('quality',{}).get('value',0)<.7 or labels.get('fails_task',{}).get('value',1)>0:reason='quality_or_task_failure'
                elif any(labels.get(k,{}).get('value',1)>.25 for k in ['toxicity','violence','not_appropriate']):reason='other_source_flags'
            if reason:rejected[reason]+=1;continue
            if any(w in norm(q['text']) for w in exclusions) or re.search(r'open.?assistant|chatgpt|openai',a['text'],re.I):rejected['manual_issue_or_identity']+=1;continue
            p,answer=q['text'].strip(),a['text'].strip()
            if norm(p) in seen or norm(answer) in answers:rejected['normalized_duplicate']+=1;continue
            group='oasst1/'+tree['message_tree_id'];bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
            split='test' if bucket<10 else 'val' if bucket<20 else 'train'
            r={'id':'oasst1/'+a['message_id'],'group':group,'domain':'human_'+q['lang'],'prompt':p,'answer':answer,'source':'OpenAssistant/oasst1; Apache-2.0; human metadata, root prompt + rank-0 reply','split':split,'source_message_ids':[q['message_id'],a['message_id']]}
            try:encode_record(r)
            except ValueError:rejected['over_1024_tokens']+=1;continue
            seen.add(norm(p));answers.add(norm(answer));records[split].append(r);kept.append(r)
    for s,rs in records.items():
        (OUT/(s+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf8')
    manifest={'source':json.loads((SOURCE/'source.json').read_text()),'max_sequence_tokens':1024,'split_method':'SHA256 whole tree ID, 80/10/10; old curriculum splits unchanged','counts':{s:dict(collections.Counter(r['domain'] for r in rs)) for s,rs in records.items()},'rejected':dict(rejected),'hashes':{s:sha(OUT/(s+'.jsonl')) for s in records},'limitations':['Source quality labels do not guarantee factual accuracy. Only root pairs used; no lost follow-up context.','Old test already inspected after pilot 01: now a regression set. New human test not used for checkpoint selection.','Human authorship is source metadata, not independently verified. Source release dates to April 2023.','Small corpus: this is a controlled continuation, not sufficient broad pretraining.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    (OUT/'ATTRIBUTION.md').write_text('# Data attribution\n\nOpenAssistant/oasst1, OpenAssistant contributors, Apache License 2.0.\nhttps://huggingface.co/datasets/OpenAssistant/oasst1\n\nPinned revision and file hashes: manifest.json. Original license and dataset card: ../corpus_sources/oasst1/.\nChanges: language/quality/length filtering, selected root question and highest-ranked response, duplicate removal, deterministic tree splits. Texts were not translated or rewritten. No third-party model weights were used.\n',encoding='utf8')
    print(json.dumps(manifest,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
