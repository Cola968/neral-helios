"""CPU plan/review/report; local inference only with --confirm-execution.

Requires the reviewed corpus module/data (#228), strict evaluation (#224), and
for actual collection the context/transport reworks (#223/#225). No downloads,
network provider, training, adapter creation, automatic grading or overwrites.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neral_control import evaluation as ev

BASE='Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION='c03e6d358207e414f1eca0bb1891e29f1db0e242'
SYSTEM='Be precise. State uncertainty. Never claim execution or sources without evidence.'
TOOLS=['calculator.v1','code_check.v1','project_files.v1']
SKIPPED='SKIPPED_NO_VERIFIED_ADAPTER'


def digest(value):
    return hashlib.sha256(value.encode('utf8')).hexdigest()


def load_corpus(root):
    try:
        from neral_control.corpus import validate,read_benchmark
    except ImportError as exc:
        raise ValueError('Reviewed corpus dependency #228 is required; no substitute data generated.') from exc
    evidence=validate(root)
    tasks,manifest=read_benchmark(root)
    ev._manifest(manifest)
    return tasks,manifest,evidence


def plan(root, *, model_dir=None, adapter_dir=None, track='resource_equivalent'):
    if track not in {'resource_equivalent','product'}:raise ValueError('Unknown comparison track')
    tasks,manifest,evidence=load_corpus(root)
    if evidence['leakage']['flagged_pairs']:raise ValueError('Corpus leakage review required')
    base=None;adapter=None
    if model_dir is not None:
        from model_catalog import validate_hf_snapshot
        base=validate_hf_snapshot(Path(model_dir).absolute(),verify_hashes=True)
    if adapter_dir is not None and base is not None:
        from neral_adapter_candidate import validate_adapter
        try:adapter=validate_adapter(Path(adapter_dir).absolute(),base)
        except (ValueError,OSError):adapter=None
    adapter_hash=digest(json.dumps(adapter,sort_keys=True)) if adapter else None
    common=dict(base_model_id=BASE,base_revision=base['revision'] if base else REVISION,
        quantization='nf4-double-bf16',seed=42,decoding={'temperature':0.,'top_k':40},
        system_prompt=SYSTEM,budgets={'max_new_tokens_per_call':256,'seconds_per_task':20})
    conditions={}
    for name in 'ABCD':
        setting=dict(deepcopy(common),tools=[] if name=='A' else TOOLS.copy(),
            adapter_sha256=adapter_hash if name in 'CD' else None,
            lora_adapter_id=adapter_hash if name in 'CD' else None,
            lora_weights_version=adapter['version'] if adapter and name in 'CD' else None,
            adapter_hash=adapter_hash if name in 'CD' else None,
            adapter_integrity_verified=bool(adapter and name in 'CD'),
            controller_memory=name=='D',controller_active=name=='D',context_control_active=name=='D')
        status=SKIPPED if name in 'CD' and adapter is None else 'PLANNED_REQUIRES_EXPLICIT_EXECUTION'
        if status!=SKIPPED:ev.validate_condition_settings(name,setting)
        calls=1 if track=='resource_equivalent' or name=='A' else 4 if name=='D' else 2
        conditions[name]=dict(status=status,settings=setting,
            execution_budget={'model_calls':calls,'tool_calls':0 if name=='A' else calls,
                              'output_tokens':256*calls,'seconds':20})
    return dict(schema=1,comparison_contract_version=2,track=track,manifest=manifest,
        tasks=len(tasks),corpus_evidence=evidence,base_snapshot_verified=base is not None,
        conditions=conditions,model_started=False,training_executed=False,
        dependencies=['reviewed corpus #228','strict evaluation #224','context #223','transport #225'],
        measured_resource_equivalence_proven=False)


class _SharedSystem:
    def __init__(self,provider,system):
        self.provider,self.system,self.identity=provider,system,provider.identity
        self.deadline=float('inf')
    def rows(self,messages):return [{'role':'system','content':self.system},*messages[1:]]
    def count_messages(self,messages,tools):return self.provider.count_messages(self.rows(messages),tools)
    def complete(self,messages,tools,options,event,on_progress=None):
        options={**options,'deadline':min(options.get('deadline',float('inf')),self.deadline)}
        return self.provider.complete(self.rows(messages),tools,options,event,on_progress)


class _LocalBackend:
    execution_kind='local_model_inference'
    def __init__(self,name,setting,*,model_dir,adapter_dir=None,allow_unvalidated_adapter=False):
        # Fail before importing torch/loading weights on an incomplete stack.
        from neral_control.context import _history_completed
        from neral_control.providers import _GenerationStop
        if _history_completed({'role':'assistant','content':'unverified'}):
            raise ValueError('Strict context dependency #223 is required')
        if model_dir is None:raise ValueError('Explicit verified local model directory is required')
        import qwen_local_runtime as local
        self.runtime=None
        try:
            if name in 'CD':
                from neral_adapter_candidate import load_candidate
                self.runtime=load_candidate(model_dir,adapter_dir,allow_unvalidated=allow_unvalidated_adapter)
            else:self.runtime=local.load(model_dir)
            from neral_control.providers import LocalQwenProvider
            provider=LocalQwenProvider(self.runtime,model_folder=model_dir,
                adapter_folder=adapter_dir if name in 'CD' else None)
            identity=provider.identity
            if (identity['base_model_revision']!=setting['base_revision']
                    or identity['lora_adapter_id']!=setting['lora_adapter_id']
                    or identity['adapter_integrity_verified']!=setting['adapter_integrity_verified']):
                raise ValueError('Loaded runtime differs from verified planned artifacts; no base fallback credit')
            self.identity=identity
            self.provider=_SharedSystem(provider,setting['system_prompt'])
        except Exception:
            self.close();raise

    def close(self):
        if self.runtime is not None:self.runtime.close();self.runtime=None

    def run(self,task,name,setting,limits):
        self.trace=dict(text='',completion_status='failed',tool_provenance=[],
                        model_calls=0,input_tokens=None,output_tokens=None,resources={})
        import tempfile
        import threading
        import torch
        from neral_control.context import ProjectReader
        from neral_control.registry import default_registry,ToolContext
        from neral_control.providers import _GenerationStop
        from neral_control.controller import ReasoningController
        import qwen_local_runtime as local
        torch.manual_seed(setting['seed']);torch.cuda.reset_peak_memory_stats(0)
        event=threading.Event();deadline=time.monotonic()+limits['seconds']
        self.provider.deadline=deadline
        stop=_GenerationStop(event,{'deadline':deadline})
        records=[];input_tokens=0;output_tokens=0;calls=0;provenance=[]
        def pack(text,status):
            from security_guard import sanitize_output
            text,_=sanitize_output(text)
            return dict(text=text,completion_status=status,tool_provenance=provenance,
                model_calls=calls,input_tokens=input_tokens,output_tokens=output_tokens,
                provenance=self.identity,resources={'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(0),
                    'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(0)})
        if name=='A':
            # Plain base generation: no tools, adapter, routing or controller.
            messages=[{'role':'system','content':setting['system_prompt']},
                      {'role':'user','content':task['prompt']}]
            ids=self.runtime.tokenizer.apply_chat_template(messages,chat_template=local.CHAT_TEMPLATE,
                tokenize=True,add_generation_prompt=True)
            input_tokens=len(ids);calls=1;self.trace.update(model_calls=1,input_tokens=input_tokens)
            tokens=self.runtime.generate(ids,256,0.,40,stop);output_tokens=len(tokens)
            status='timed_out' if stop.is_set() else 'complete' if self.runtime.finish_reason=='eos' else 'incomplete'
            return pack(self.runtime.decode(tokens),status)
        with tempfile.TemporaryDirectory(prefix='neral-eval-fixture-') as folder:
            for source in task.get('project_files',[]):
                relative=source['relative_path'];ProjectReader._parts(relative)
                if digest(source['content'])!=source['sha256']:raise ValueError('Project fixture hash mismatch')
                path=Path(folder)/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(source['content'],encoding='utf8')
            reader=ProjectReader(folder);registry=default_registry(reader)
            permissions=frozenset({'math','code_check','project_read'})
            if name=='D':
                # Extra reasoning/system overhead is a declared treatment; replace
                # its persona with the same common system through the provider.
                for source in task.get('project_files',[])[:3]:records.append(reader.read(source['relative_path']))
                self.trace.update(model_calls=None,tool_provenance=None)
                result=ReasoningController(registry,self.provider).run(task['prompt'],
                    profile='fast' if limits['model_calls']==1 else 'deep',
                    allowed_tools=('calculator','code_check','project_files'),permissions=permissions,
                    event=event,max_new_tokens=256,project_records=records)
                calls=result['model_calls'];input_tokens=result['tokens_used']['input'];output_tokens=result['tokens_used']['output']
                provenance=[{'tool_id':r['tool_id']+'.v'+r['version'],'call_index':i,'status':r['status']}
                            for i,r in enumerate(result['tool_results'])]
                return pack(result['text'],result.get('completion_status','failed'))
            messages=[{'role':'system','content':setting['system_prompt']},{'role':'user','content':task['prompt']}]
            schemas=registry.schemas(('calculator','code_check','project_files'));seen=set()
            for _ in range(limits['model_calls']):
                calls+=1;self.trace['model_calls']=calls
                reply=self.provider.complete(messages,schemas,dict(max_new_tokens=256,temperature=0.,
                    top_k=40,profile='standard',native_think=False,deadline=deadline),event)
                input_tokens=None if input_tokens is None or reply.input_tokens is None else input_tokens+reply.input_tokens
                output_tokens=None if output_tokens is None or not reply.output_tokens_reported else output_tokens+reply.generated_tokens
                self.trace.update(input_tokens=input_tokens,output_tokens=output_tokens)
                if reply.completion_status!='complete':return pack(reply.content,reply.completion_status)
                if not reply.tool_calls:return pack(reply.content,'complete')
                if len(reply.tool_calls)!=1 or len(provenance)>=limits['tool_calls']:
                    return pack('Tool budget exceeded.','failed')
                fn=reply.tool_calls[0]['function'];signature=ev._canonical(fn)
                if signature in seen:return pack('Repeated tool call rejected.','failed')
                seen.add(signature)
                if fn['name'] not in {'calculator','code_check','project_files'}:
                    return pack('Unapproved tool rejected.','failed')
                provenance.append({'tool_id':fn['name']+'.v1','call_index':len(provenance),'status':'error'})
                self.trace['tool_provenance']=provenance
                result=registry.execute(fn['name'],fn['arguments'],ToolContext(permissions,event,deadline))
                provenance[-1]['status']=result['status']
                if calls==limits['model_calls']:
                    return pack(json.dumps(result.get('output'),ensure_ascii=False),
                                'complete' if result['status']=='ok' else 'failed')
                messages.extend([{'role':'assistant','content':reply.content,'tool_calls':reply.tool_calls},
                    {'role':'tool','tool_name':fn['name'],'content':json.dumps(result,ensure_ascii=False)}])
            return pack('Generation budget exhausted.','incomplete')


def collect(root, *, confirm_execution=False, conditions=tuple('ABCD'), backend_factory=None,
            model_dir=None,adapter_dir=None,track='resource_equivalent',allow_unvalidated_adapter=False):
    if confirm_execution is not True:raise PermissionError('--confirm-execution is required before any collection')
    if (not conditions or any(c not in ('A','B','C','D') for c in conditions)
            or len(set(conditions))!=len(conditions)):raise ValueError('Invalid conditions')
    prepared=plan(root,model_dir=model_dir,adapter_dir=adapter_dir,track=track)
    tasks,manifest,_=load_corpus(root);result={};kind=None
    if ev._canonical(manifest)!=ev._canonical(prepared['manifest']):
        raise ValueError('Corpus changed after preflight')
    factory=backend_factory or _LocalBackend
    for name in conditions:
        condition=prepared['conditions'][name];rows=[]
        if condition['status']==SKIPPED:
            result[name]={'status':SKIPPED,'results':[]};continue
        backend=None
        try:
            try:
                backend=factory(name,condition['settings'],model_dir=model_dir,adapter_dir=adapter_dir,
                                allow_unvalidated_adapter=allow_unvalidated_adapter)
            except Exception:
                result[name]={'status':SKIPPED if name in 'CD' else 'FAILED_RUNTIME_UNAVAILABLE',
                              'results':[],'reason':'runtime_or_adapter_activation_failed'}
                continue
            kind=getattr(backend,'execution_kind','unverified_backend')
            for task in tasks:
                started=time.monotonic()
                try:row=backend.run(task,name,condition['settings'],condition['execution_budget'])
                except Exception:
                    # Keep the attempted case. Never record private exception text.
                    row=deepcopy(getattr(backend,'trace',dict(text='',completion_status='failed',
                        tool_provenance=None,model_calls=None,input_tokens=None,output_tokens=None,resources={})))
                row.update(id=task['id'],prompt_hash=task['prompt_hash'],rubric_hash=task['rubric_hash'],
                    latency_ms=(time.monotonic()-started)*1000,enabled_tools=condition['settings']['tools'])
                row['answer_sha256']=digest(row['text'])
                row['tool_calls']=len(row['tool_provenance']) if row['tool_provenance'] is not None else None
                row['used_tools']=sorted({p['tool_id'] for p in row['tool_provenance']}) if row['tool_provenance'] is not None else None
                rows.append(row)
            result[name]={'status':'COLLECTED','results':rows}
        finally:
            if backend is not None:backend.close()
    return dict(schema=1,plan=prepared,conditions=result,execution_kind=kind,
                training_executed=False,model_quality_gain_measured=False)


def score(collection,reviews):
    prepared=collection['plan'];manifest=prepared['manifest'];expected=ev._manifest(manifest)
    if not isinstance(reviews,dict):raise ValueError('Reviews must map collected conditions')
    required={n for n,c in collection['conditions'].items() if c['status']=='COLLECTED'}
    if set(reviews)!=required:raise ValueError('Review all collected conditions, no skipped/synthetic replacements')
    runs={};summaries={};indexed={}
    tasks={r['id']:r for r in manifest['tasks']}
    for name,condition in collection['conditions'].items():
        if condition['status']!= 'COLLECTED':summaries[name]={'status':condition['status']};continue
        if not isinstance(reviews[name],list) or len(reviews[name])!=len(expected):raise ValueError('Review inventory mismatch')
        assessed={r['id']:r for r in reviews[name]}
        if set(assessed)!=expected:raise ValueError('Duplicate or missing task review')
        setting=prepared['conditions'][name]['settings'];ev.validate_condition_settings(name,setting)
        rows=[]
        for output in condition['results']:
            review=assessed[output['id']];task=tasks[output['id']]
            if (set(review)!={'id','status','unsupported_claims','failure_causes','reviewer',
                              'prompt_hash','rubric_hash','answer_sha256'}
                    or not ev._text(review['reviewer'],200)
                    or review['prompt_hash']!=task['prompt_sha256'] or review['rubric_hash']!=task['rubric_sha256']
                    or review['answer_sha256']!=digest(output['text'])
                    or output['answer_sha256']!=review['answer_sha256']
                    or output['prompt_hash']!=review['prompt_hash'] or output['rubric_hash']!=review['rubric_hash']):
                raise ValueError('Review is not bound to the frozen rubric and exact output')
            row={k:output[k] for k in ('id','completion_status','used_tools','enabled_tools','tool_provenance',
                'tool_calls','input_tokens','output_tokens','latency_ms','resources')}
            row.update(status=review['status'],complete=output['completion_status']=='complete',
                       unsupported_claims=review['unsupported_claims'],failure_causes=review['failure_causes'])
            rows.append(row)
        indexed[name]=ev._results(rows,expected,setting)
        summaries[name]={'status':'SCORED',**ev._summary(indexed[name])}
        runs[name]={'settings':setting,'results':rows}
    if set(runs)==set('ABCD'):
        report=ev.evaluate(manifest,runs);summaries={n:{'status':'SCORED',**report['conditions'][n]} for n in 'ABCD'}
    else:
        denominator=sum(r['effective_status']!='full' for r in indexed.get('A',{}).values()) if 'A' in indexed else None
        report={'target':{'numerator':None,'denominator':denominator,'rate':None,'achieved':False},
                'regressions_D_vs_A':None,'independent_quality_proven':False,'release_authorized':False}
    for name in 'ABCD':summaries.setdefault(name,{'status':'NOT_REQUESTED'})
    report.update(conditions=summaries,benchmark_version=manifest['benchmark_version'],cases=len(expected),
        manifest_sha256=digest(ev._canonical(manifest)),collection_sha256=digest(ev._canonical(collection)),
        comparison_track=prepared['track'],base_model_id=BASE,base_revision=prepared['conditions']['A']['settings']['base_revision'],
        adapter_hash=prepared['conditions']['D']['settings']['adapter_hash'],
        execution_budgets={n:c['execution_budget'] for n,c in prepared['conditions'].items()},
        measured_resource_equivalence_proven=False,model_quality_gain_measured=False,
        execution_kind=collection.get('execution_kind'),independent_quality_proven=False,
        limitations=['Imported execution/review assertions are not independently authenticated.',
                    'DEV tasks are known, not unseen holdouts. No learned-quality or release claim.',
                    'Resource track matches maximum calls/output/time; tool/template input overhead differs.'])
    for name in required:
        measured=[r['model_calls'] for r in collection['conditions'][name]['results'] if r['model_calls'] is not None]
        report['conditions'][name]['metrics']['model_calls']=ev._measurement(measured)
    report['target']['percent']=None if report['target']['rate'] is None else 100*report['target']['rate']
    return report


def markdown(report):
    target=report['target']
    lines=['# NERAL A/B/C/D DEV report','',f"Benchmark: {report['benchmark_version']} ({report['cases']} cases)",
        f"Track: {report['comparison_track']}",f"Base: {report['base_model_id']} @ {report['base_revision']}",
        f"Adapter hash: {report['adapter_hash'] or 'none verified'}",'',
        '| Condition | State | Full | Partial | Fail |','|---|---|---:|---:|---:|']
    for name,c in report['conditions'].items():
        lines.append(f"| {name} | {c['status']} | {c.get('full','—')} | {c.get('partial','—')} | {c.get('fail','—')} |")
    lines.extend(['','| Condition | Model calls | Tool calls | Input tokens | Output tokens | Mean ms | Peak allocated bytes |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for name,c in report['conditions'].items():
        metrics=c.get('metrics',{})
        def measured(key):
            item=metrics.get(key,{})
            return f"{item.get('total','unknown')} ({item.get('measured_cases',0)} measured)"
        lines.append(f"| {name} | {measured('model_calls')} | {measured('tool_calls')} | {measured('input_tokens')} | {measured('output_tokens')} | {metrics.get('latency_ms',{}).get('mean','unknown')} | {metrics.get('resources',{}).get('peak_cuda_allocated_bytes',{}).get('max','unknown')} |")
    reduction=(f"{target['numerator']} / {target['denominator']}; {target['percent']}%"
               if target['rate'] is not None else f"not evaluated (baseline failures: {target['denominator']})")
    regressions=report.get('regressions_D_vs_A')
    lines.extend(['',f"Error reduction: {reduction}.",
        f"Regressions D vs A: {len(regressions) if regressions is not None else 'not evaluated'}; {regressions}",
        f"Execution budgets: {json.dumps(report['execution_budgets'],sort_keys=True)}",
        f"Manifest SHA-256: {report['manifest_sha256']}",f"Collection SHA-256: {report['collection_sha256']}",
        'No independent model-quality, learned-weight or release proof. Skipped conditions contain no simulated responses.',''])
    return '\n'.join(lines)


def read(path):
    if path.stat().st_size>32*1024**2:raise ValueError('Input exceeds 32 MiB')
    return json.loads(path.read_text(encoding='utf8'),object_pairs_hook=ev._unique_keys,
                      parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write(path,text):
    with path.open('x',encoding='utf8',newline='\n') as stream:stream.write(text)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('plan','collect','score','report'))
    parser.add_argument('--corpus-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model-dir',type=Path);parser.add_argument('--adapter-dir',type=Path)
    parser.add_argument('--track',choices=('resource_equivalent','product'),default='resource_equivalent')
    parser.add_argument('--conditions',nargs='+',choices=list('ABCD'),default=list('ABCD'))
    parser.add_argument('--confirm-execution',action='store_true')
    parser.add_argument('--allow-unvalidated-adapter',action='store_true')
    parser.add_argument('--collection',type=Path);parser.add_argument('--reviews',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--markdown-output',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.output and args.markdown_output and args.output.resolve()==args.markdown_output.resolve():
            raise ValueError('JSON and Markdown need distinct output paths')
        for path in (args.output,args.markdown_output):
            if path is not None and (path.exists() or path.is_symlink()):
                raise ValueError('Output already exists; no execution or overwrite')
            if path is not None and not path.parent.is_dir():
                raise ValueError('Output parent must already exist')
        if args.mode in {'plan','collect'}:
            options=dict(model_dir=args.model_dir,adapter_dir=args.adapter_dir,track=args.track)
            value=(plan(args.corpus_root,**options) if args.mode=='plan' else collect(args.corpus_root,
                confirm_execution=args.confirm_execution,conditions=tuple(args.conditions),
                allow_unvalidated_adapter=args.allow_unvalidated_adapter,**options))
        else:
            if not args.collection or not args.reviews:raise ValueError('Collection and bound reviews required')
            value=score(read(args.collection),read(args.reviews))
            if args.mode=='report':
                if not args.output or not args.markdown_output:raise ValueError('Report requires both JSON and Markdown outputs')
                write(args.markdown_output,markdown(value))
        rendered=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        if args.output:write(args.output,rendered)
        else:print(rendered,end='')
    except (ValueError,OSError,PermissionError,KeyError,TypeError,ImportError) as exc:
        parser.exit(2,f'Runner refused input/execution: {type(exc).__name__}. No automatic retry or download.\n')


if __name__=='__main__':main()
