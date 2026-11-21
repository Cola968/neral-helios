"""Owner-only real local HTTP/CSRF smoke of the exact-note route; no GPU inference."""
from __future__ import annotations
import argparse
import http.cookiejar
import json
from pathlib import Path
from urllib.request import Request, build_opener, HTTPCookieProcessor, ProxyHandler

BASE='http://127.0.0.1:9690'

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    args=p.parse_args(argv)
    rows=[json.loads(x) for x in args.data.read_text(encoding='utf8').splitlines() if x.strip()]
    item=next(x for x in rows if x['meta'].get('category')=='mathematics' and
              x['meta'].get('source')=='v39_benchmark')
    opener=build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()),ProxyHandler({}))
    with opener.open(BASE+'/',timeout=8) as response:
        if response.status!=200:raise RuntimeError('Local session creation failed')
    with opener.open(BASE+'/api/neral/preview/session',timeout=8) as response:
        state=json.load(response)
    csrf=state.get('csrf')
    if not isinstance(csrf,str) or len(csrf)<12:
        raise RuntimeError('Valid local CSRF token unavailable')
    payload=json.dumps({'tier':'Luna','prompt':item['instruction'],'history':[]},ensure_ascii=False).encode('utf8')
    req=Request(BASE+'/api/neral/chat/three-tier',data=payload,headers={
        'Content-Type':'application/json','X-Neral-CSRF':csrf,
        'Origin':BASE},method='POST')
    with opener.open(req,timeout=15) as response:
        if response.status!=200:raise RuntimeError('Local exact-memory API did not return 200')
        output=json.load(response)
    if (output.get('local_note_hit') is not True or
        output.get('text')!=item['response'] or
        output.get('note_source')!='v39_benchmark' or
        output.get('verifier_ok') is not False or
        output.get('model_generated') is not False):
        raise RuntimeError('Local HTTP memory response did not meet provenance/answer contract')
    print('NERAL_V310_EXACT_LOCAL_NOTE_HTTP_CSRF_200_VERIFIED_NO_MODEL_LOADING')
    return 0

if __name__=='__main__':raise SystemExit(main())
