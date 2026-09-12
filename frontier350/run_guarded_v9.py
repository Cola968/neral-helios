"""Bounded V9 A/B and one confirmation segment, with unchanged strict capability floors."""
import argparse,json,math,os,shutil,subprocess,sys
from pathlib import Path
from recovery_identity import file_sha
from interpolate_recovery_checkpoint import interpolate

ANCHOR_SHA='4a2e2682fa215b02132f552bc3cdf251abbbfc5944cd4631d6f9065c03583eab'
ANCHOR_TOKENS=1_004_003_584


def strict_bilingual(base,candidate,dev):
    for obj in (base,candidate):
        if obj.get('cases')!=400 or set(obj.get('directions',{}))!={'de_to_en','en_to_de'}:
            raise ValueError('incomplete bilingual evidence')
        total=correct=0
        for direction in obj['directions'].values():
            n,k=direction['total'],direction['correct']
            if type(n)is not int or type(k)is not int or n!=200 or not 0<=k<=n:
                raise ValueError('invalid bilingual counts')
            if isinstance(direction['accuracy_pct'],bool) or not math.isfinite(direction['accuracy_pct']):
                raise ValueError('invalid bilingual percentage')
            if abs(direction['accuracy_pct']-100*k/n)>1e-9:raise ValueError('inconsistent direction score')
            total+=n;correct+=k
        if isinstance(obj['overall_accuracy_pct'],bool) or not math.isfinite(obj['overall_accuracy_pct']) or abs(obj['overall_accuracy_pct']-100*correct/total)>1e-9:raise ValueError('inconsistent total score')
    for key in ('cases_sha256','tokenizer_sha256','protocol_sha256'):
        if not base.get(key) or base[key]!=candidate.get(key):raise ValueError('bilingual protocol/data changed')
    for key in ('checkpoint_sha256','checkpoint_processed_tokens','tokenizer_sha256'):
        if candidate[key]!=dev[key]:raise ValueError('mixed candidate checkpoint evidence')
    return candidate['overall_accuracy_pct']>=base['overall_accuracy_pct'] and all(
        candidate['directions'][d]['accuracy_pct']>=base['directions'][d]['accuracy_pct'] for d in base['directions'])


def main():
    ap=argparse.ArgumentParser()
    for name in ('assets','report','request-id'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();assets=Path(a.assets).resolve();report=Path(a.report).resolve();report.mkdir(parents=True,exist_ok=True)
    anchor=assets/'runs/neral350_safe_v8_arctus-safe-v8-20260912-r1_selected/resume.pt'
    if file_sha(anchor)!=ANCHOR_SHA:raise ValueError('accepted V8 anchor changed')
    tok=assets/'frontier350_assets/tokenizer.json'
    main_data=assets/'frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05'
    replay=assets/'frontier350_replay_v9_20260912'
    valid=assets/'frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid'
    cases=main_data/'bilingual_holdout.json'
    for name,path in (('replay_manifest',replay/'manifest.json'),('main_manifest',main_data/'manifest.json')):
        shutil.copyfile(path,report/(name+'.json'))
    run_root=assets/'runs'/('neral350_v9_'+a.request_id)
    if run_root.exists():raise FileExistsError('V9 run already exists; inspect resume state, never overwrite')
    run_root.mkdir()
    def execute(script,*args):
        subprocess.run([sys.executable,'-u',str(Path(__file__).parent/script),*map(str,args)],check=True)
    def evaluate(ck,tag,tokens):
        execute('eval_validation_loss350.py','--checkpoint',ck,'--valid-data',valid,'--out',report/f'{tag}_validation.json',
                '--seq',512,'--batch',4,'--batches',32,'--seed',351337)
        execute('eval_recovery_dev_v2.py','--checkpoint',ck,'--tokenizer',tok,'--out',report/f'{tag}_dev.json','--expected-tokens',tokens)
        execute('eval_bilingual_holdout_v1.py','--checkpoint',ck,'--tokenizer',tok,'--cases',cases,
                '--out',report/f'{tag}_bilingual.json','--expected-tokens',tokens)
        return {key:json.loads((report/f'{tag}_{key}.json').read_text()) for key in ('dev','validation','bilingual')}
    baseline=evaluate(anchor,'v8_baseline',ANCHOR_TOKENS)
    records=[];pilots={}
    def train(source,mode,tag,probe=False):
        out=run_root/tag
        command=['--anchor',source,'--anchor-sha',file_sha(source),'--main-data',main_data/'train',
                 '--replay',replay,'--out',out,'--mode',mode,'--new-tokens',2_000_000]
        if probe:
            execute('train_guarded_v9.py',*command,'--max-updates',2)
            smoke=json.loads((out/'summary.json').read_text());shutil.copyfile(out/'summary.json',report/f'{tag}_gpu_probe.json')
            if smoke['new_tokens']!=49_152 or smoke['peak_allocated_bytes']>10*1024**3:
                raise RuntimeError('GPU probe outside validated memory/token budget; weights retained')
            execute('train_guarded_v9.py',*command,'--resume',out/'resume.pt')
        else:execute('train_guarded_v9.py',*command)
        summary=json.loads((out/'summary.json').read_text())
        if not summary['complete'] or summary['new_tokens']!=2_000_000:raise ValueError('pilot budget incomplete')
        for name in ('summary.json','metrics.jsonl'):shutil.copyfile(out/name,report/f'{tag}_{name}')
        return out/'resume.pt',summary
    def gate(source,candidate,base,base_tag,tag,tokens):
        for alpha in (1.,.4,.1):
            label=f'{tag}_a{round(alpha*100):03d}';path=run_root/(label+'.pt')
            interpolate(source,candidate,path,alpha)
            result=evaluate(path,label,tokens)
            gate_path=report/f'{label}_gate.json'
            execute('select_recovery_lr_pilot.py','--baseline-dev',report/f'{base_tag}_dev.json',
                    '--baseline-validation',report/f'{base_tag}_validation.json',
                    '--candidate',label,7.5e-5,report/f'{label}_dev.json',report/f'{label}_validation.json','--out',gate_path)
            legacy=json.loads(gate_path.read_text())
            bilingual=strict_bilingual(base['bilingual'],result['bilingual'],result['dev'])
            improved=result['validation']['validation_loss']<base['validation']['validation_loss'] or result['bilingual']['overall_accuracy_pct']>base['bilingual']['overall_accuracy_pct'] or any(result['dev']['categories'][c]['accuracy_pct']>base['dev']['categories'][c]['accuracy_pct'] for c in base['dev']['categories'])
            passed=legacy['selection_allowed'] and bilingual and improved
            row={'tag':label,'alpha':alpha,'legacy_gate':legacy['selection_allowed'],'bilingual_gate':bilingual,
                 'strict_improvement':improved,'pass':passed,'validation_loss':result['validation']['validation_loss'],
                 'bilingual_pct':result['bilingual']['overall_accuracy_pct'],'checkpoint_sha256':file_sha(path)}
            records.append(row);(report/'progress.json').write_text(json.dumps(records,indent=2))
            if passed:return {'path':path,'tag':label,'result':result,'row':row}
            # Only this run's rejected temporary evaluation snapshot is removed.
            if path.parent.resolve()!=run_root.resolve():raise ValueError('unsafe snapshot path')
            path.unlink()
        return None
    # Probe and continue each SAME pilot; don't throw away GPU warmup weights.
    for mode in ('replay_only','projected'):
        candidate,summary=train(anchor,mode,mode,probe=True)
        pilots[mode]={'summary':summary,'accepted':gate(anchor,candidate,baseline,'v8_baseline',mode,ANCHOR_TOKENS+2_000_000)}
    if pilots['replay_only']['summary']['batch_chain']!=pilots['projected']['summary']['batch_chain']:
        raise ValueError('A/B input batches differ; selection refused')
    accepted=[(mode,p['accepted']) for mode,p in pilots.items() if p['accepted']]
    selected=None;final_tokens=ANCHOR_TOKENS;confirmed=False
    if accepted:
        mode,selected=min(accepted,key=lambda x:(x[1]['row']['validation_loss'],-x[1]['row']['bilingual_pct']))
        final_tokens+=2_000_000
        candidate,_=train(selected['path'],mode,'confirmation')
        confirmation=gate(selected['path'],candidate,selected['result'],selected['tag'],'confirmation',final_tokens+2_000_000)
        if confirmation:selected=confirmation;final_tokens+=2_000_000;confirmed=True
    result={'schema':1,'start_tokens':ANCHOR_TOKENS,'final_tokens':final_tokens,'confirmation_passed':confirmed,
            'selected':str(selected['path']) if selected else None,'selected_sha256':file_sha(selected['path']) if selected else ANCHOR_SHA,
            'records':records,'promoted_to_chat':False,'final_audit_pending':True,
            'note':'The 400-case bilingual suite is a repeatedly used development gate, not an untouched release holdout.'}
    (report/'selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
