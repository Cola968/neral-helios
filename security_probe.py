import argparse,json
from pathlib import Path
from security_guard import check_prompt,sanitize_output
from security_cases import BLOCK,ALLOW

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();cases=[]
    for allowed,examples in [(False,BLOCK),(True,ALLOW)]:
        for prompt in examples:
            result=check_prompt(prompt);cases.append({'prompt':prompt,'expected_allowed':allowed,'actual_allowed':result.allowed,'category':result.category,'passed':result.allowed==allowed})
    import app as backend
    backend.app.config['TESTING']=False
    client=backend.app.test_client()
    tests=[('json_array',dict(json=['not','an','object']),400),('non_string_prompt',dict(json={'prompt':42}),400),('non_finite_temperature',dict(json={'prompt':'Hallo','temperature':float('inf')}),400),('bad_host',dict(json={'prompt':'Hallo'},headers={'Host':'evil.example'}),403),('bad_origin',dict(json={'prompt':'Hallo'},headers={'Origin':'https://evil.example'}),403),('cross_site',dict(json={'prompt':'Hallo'},headers={'Sec-Fetch-Site':'cross-site'}),403),('non_json',dict(data='prompt=Hallo'),415),('large_prompt',dict(json={'prompt':'a'*5000}),413)]
    # Avoid generating during API input validation probes. Invalid requests must not reach inference.
    original=backend.model.generate
    def denied_generation(*a,**k):raise RuntimeError('INVALID REQUEST REACHED INFERENCE')
    backend.model.generate=denied_generation
    api=[]
    try:
        for name,kw,status in tests:
            backend._hits.clear();resp=client.post('/api/generate',base_url='http://127.0.0.1:9680',**kw);api.append({'name':name,'expected':status,'actual':resp.status_code,'passed':resp.status_code==status})
    finally:backend.model.generate=original
    report={'guard':{'passed':sum(c['passed'] for c in cases),'total':len(cases),'cases':cases},'api':{'passed':sum(c['passed'] for c in api),'total':len(api),'cases':api},'scope':'Curated regression cases, not comprehensive adversarial safety evaluation.'}
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:{a:b for a,b in v.items() if a!='cases'} for k,v in report.items() if isinstance(v,dict)},indent=2))
if __name__=='__main__':main()
