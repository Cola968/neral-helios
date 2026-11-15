"""Manual RTX acceptance through the actual opt-in THINK Flask handler.

The Flask transport is in-process; ALL model replies come from real localhost
Ollama calls. The recorder forwards requests unchanged and never substitutes
fixtures. Only three public historical cases plus an added-clause negative
are used. No sealed dataset access, production defaults, training or upload.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not args.live: parser.error('Explicit --live required')
    args.out.mkdir(parents=True,exist_ok=False)
    from flask import Flask
    from neral_agent_autonomy import OllamaLocalBackend, local_only_opener
    from neral_local_chat_api import register_neral_local_chat_api
    import neral_nl_local_ollama as local
    from scripts.neral_nl_schema_probe import old_case_question

    def info(path):
        with local_only_opener().open('http://127.0.0.1:11434'+path,timeout=10) as response:
            return json.loads(response.read(131072))
    def record_residency(record):
        # Monitoring must never change the underlying inference result.
        try: record['gpu_residency_after_response']=info('/api/ps')
        except Exception as exc: record['gpu_monitor_error_class']=type(exc).__name__
    inventory=info('/api/tags')
    names={m['name'] for m in inventory['models']}
    if not {'neral-qwen35-qlora-v1:latest','qwen3.5:9b'}.issubset(names):
        raise RuntimeError('Both exact local models must already exist')
    os.environ['NERAL_EXPERIMENTAL_NL_SCHEMA']='1'
    backend=OllamaLocalBackend('qwen3.5:9b',inference_profile='measured_nonthinking_2048')
    model_records=[]; fallback_records=[]
    actual_transport=local._localhost_transport
    def record_transport(packet):
        started=time.perf_counter()
        record={'model':packet['model'],'constraints_budget':{
            k:packet['format']['properties']['constraints'].get(k)
            for k in ('minItems','maxItems','uniqueItems')}}
        model_records.append(record)
        try:
            response=actual_transport(packet)
            record.update(raw_json=response.get('message',{}).get('content'),
                          done_reason=response.get('done_reason'),
                          eval_count=response.get('eval_count'),
                          prompt_eval_count=response.get('prompt_eval_count'))
            for key in ('load_duration','eval_duration','prompt_eval_duration','total_duration'):
                value=response.get(key)
                record[key+'_ms']=value/1_000_000 if type(value) is int else None
            record_residency(record)
            return response
        except Exception as exc:
            record['error_class']=type(exc).__name__
            raise
        finally: record['elapsed_ms']=(time.perf_counter()-started)*1000
    actual_fallback=backend.respond_chat
    def record_fallback(turns):
        started=time.perf_counter(); record={}; fallback_records.append(record)
        try:
            result=actual_fallback(turns)
            record['response']=result
            record_residency(record)
            return result
        except Exception as exc:
            record.update(error_class=type(exc).__name__,reason=getattr(exc,'reason',None))
            raise
        finally: record['elapsed_ms']=(time.perf_counter()-started)*1000
    backend.respond_chat=record_fallback
    app=Flask('neral-pr542-real-local-acceptance')
    register_neral_local_chat_api(app,backend)
    cases=[(case,old_case_question(case)) for case in ('21453','logic-02','logic-06')]
    cases.append(('negative-extra-clause',old_case_question('21453')+
                  ' Außerdem steht die Ziffer 1 ganz links.'))
    records=[]
    local._localhost_transport=record_transport
    try:
        for case,question in cases:
            model_records.clear(); fallback_records.clear()
            started=time.perf_counter()
            response=app.test_client().post('/api/neral/chat/think',json={
                'prompt':question,'nl_schema_consensus':True})
            record={'case':case,'http_status':response.status_code,
                    'elapsed_ms':(time.perf_counter()-started)*1000,
                    'result':response.get_json(),'model_calls':list(model_records),
                    'raw_base_calls':list(fallback_records)}
            (args.out/(case+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            records.append(record)
            print(json.dumps({'case':case,'http_status':response.status_code,
                'answer':record['result'].get('final_answer'),
                'safe_to_promote':record['result'].get('safe_to_promote'),
                'selected_extractor':record['result'].get('selected_extractor'),
                'model_calls':len(model_records),'raw_base_calls':len(fallback_records),
                'elapsed_ms':record['elapsed_ms']},ensure_ascii=False),flush=True)
    finally: local._localhost_transport=actual_transport
    summary={'transport':'Flask THINK handler, real localhost Ollama passthrough',
             'models':inventory,'cases':[
        {'case':r['case'],'http_status':r['http_status'],'result':r['result'],
         'elapsed_ms':r['elapsed_ms'],'model_calls':len(r['model_calls']),
         'raw_base_calls':len(r['raw_base_calls'])} for r in records]}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__': main()
