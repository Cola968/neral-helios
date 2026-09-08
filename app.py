"""Local Neral/Helios browser test: own unchanged 20M architecture."""
from __future__ import annotations
import hashlib,json,math,os,re,secrets,threading,time
from collections import defaultdict,deque
from pathlib import Path
import torch
from flask import Flask,jsonify,render_template,request
from werkzeug.exceptions import HTTPException
from config import NeralConfig
from model import Neral20M
from tokenizer import ByteTokenizer
from security_guard import check_prompt,sanitize_output
from solve import solve
from code_check import check_code
from architectures import expected_parameters
from project_store import ProjectStore,ripple

BASE=Path(__file__).resolve().parent
# Deployment configuration only; HTTP clients cannot choose filesystem paths.
ASSETS=Path(os.environ.get("HELIOS_ASSETS_ROOT",str(BASE))).resolve()
EXPECTED_PARAMS=20_587_008
MAX_PROMPT_BYTES=4096
MAX_NEW_TOKENS=512
APP_PORT=int(os.environ.get('HELIOS_PORT','9680'))
if not 1024<=APP_PORT<=65535:raise ValueError('HELIOS_PORT must be 1024..65535')
ALLOWED_HOSTS={f'127.0.0.1:{APP_PORT}',f'localhost:{APP_PORT}','127.0.0.1','localhost'}
ALLOWED_ORIGINS={f'http://127.0.0.1:{APP_PORT}',f'http://localhost:{APP_PORT}'}
RATE_LIMIT=30
RATE_WINDOW_S=60
app=Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=24*1024,JSON_SORT_KEYS=False)
project_store=ProjectStore(BASE/'local_state/projects.sqlite3')
_hits=defaultdict(deque);_hit_lock=threading.Lock();_generation_lock=threading.Lock();_jobs={};_job_lock=threading.Lock()
torch.set_num_threads(2);torch.manual_seed(968);torch.set_float32_matmul_precision('high')
cfg=NeralConfig();model=Neral20M(cfg);tokenizer=ByteTokenizer();report=model.parameter_report()
assert report['trainable']==EXPECTED_PARAMS
device_name=os.environ.get('HELIOS_DEVICE','cpu')
if device_name not in {'cpu','cuda'}:raise ValueError('HELIOS_DEVICE must be cpu or cuda')
device=torch.device(device_name)
runtime_error=None
checkpoint_loaded=False;checkpoint_meta={};checkpoint_sha256=None;active_checkpoint='original'

def checkpoint_paths():
    paths={'original':(ASSETS/'checkpoints/neral20m.pt',ASSETS/'checkpoints/neral20m.sha256')}
    if (ASSETS/'export/neral20m.pt').exists():paths['pilot_01']=(ASSETS/'export/neral20m.pt',ASSETS/'export/neral20m.sha256')
    if (ASSETS/'export/pilot_02/neral20m.pt').exists():paths['pilot_02']=(ASSETS/'export/pilot_02/neral20m.pt',ASSETS/'export/pilot_02/neral20m.sha256')
    if (ASSETS/'export/neral40m/neral40m.pt').exists():paths['neral40m']=(ASSETS/'export/neral40m/neral40m.pt',ASSETS/'export/neral40m/neral40m.sha256')
    if (ASSETS/'export/neral50m/neral50m.pt').exists():paths['neral50m']=(ASSETS/'export/neral50m/neral50m.pt',ASSETS/'export/neral50m/neral50m.sha256')
    return paths
def load_runtime(name):
    global checkpoint_loaded,checkpoint_meta,checkpoint_sha256,active_checkpoint,model,cfg,report,runtime_error
    if name not in checkpoint_paths():raise ValueError('Unbekannter Checkpoint.')
    path,hash_path=checkpoint_paths()[name]
    if not path.is_file() or not hash_path.is_file():raise ValueError('Checkpoint oder SHA-256-Datei fehlt.')
    if path.stat().st_size>200*1024*1024:raise ValueError('Inferenz-Checkpoint zu groß.')
    actual=hashlib.sha256(path.read_bytes()).hexdigest();hash_parts=hash_path.read_text(encoding='ascii').split();expected=hash_parts[0] if hash_parts else ''
    if not re.fullmatch('[a-fA-F0-9]{64}',expected) or not secrets.compare_digest(actual,expected.lower()):raise ValueError('Checkpoint-Integrität fehlgeschlagen.')
    payload=torch.load(path,map_location='cpu',weights_only=True);state=payload['model']
    candidate_cfg=NeralConfig(**payload.get('config',{}));expected=expected_parameters(candidate_cfg)
    if not torch.equal(state['tok_emb.weight'],state['lm_head.weight']):raise ValueError('Inkonsistente gebundene Gewichte.')
    if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Ungültige Gewichte.')
    if expected==50_012_976:
        from model50 import Neral50M
        candidate=Neral50M(candidate_cfg)
    else:candidate=Neral20M(candidate_cfg)
    candidate.load_state_dict(state,strict=True)
    if candidate.parameter_report()['total']!=expected:raise ValueError('Parameterzahl stimmt nicht.')
    candidate.to(device);candidate.eval();model=candidate;cfg=candidate_cfg;report=model.parameter_report()
    checkpoint_meta={k:v for k,v in payload.items() if isinstance(v,(str,int,float,bool))}
    checkpoint_sha256=actual;checkpoint_loaded=True;active_checkpoint=name;runtime_error=None
def initialize_runtime():
    global runtime_error
    name='pilot_02' if (ASSETS/'export/pilot_02/neral20m.pt').is_file() else 'original'
    if not checkpoint_paths()[name][0].is_file():
        runtime_error='Kein eigener Checkpoint verfügbar. SOLVE, Codeprüfung und Projekte sind bereit.'
        return
    try:load_runtime(name)
    except Exception:
        app.logger.exception('Eigener Checkpoint konnte nicht geladen werden')
        runtime_error='Checkpoint konnte nicht sicher geladen werden. SOLVE, Codeprüfung und Projekte sind bereit.'

initialize_runtime()

@app.route('/api/projects',methods=['GET','POST'])
def projects():
    if request.method=='GET':return jsonify(projects=project_store.list())
    data,error=body()
    if error:return error
    try:return jsonify(project_store.create(data.get('name'))),201
    except ValueError as e:return jsonify(error=str(e)),400

@app.get('/api/projects/<project>')
def project_detail(project):
    try:return jsonify(project_store.get(project))
    except ValueError as e:return jsonify(error=str(e)),404

@app.post('/api/projects/<project>/<action>')
def project_action(project,action):
    data,error=body()
    if error:return error
    try:
        if action=='nodes':result=project_store.add(project,data)
        elif action=='ripple':result={'affected':ripple(project_store.get(project)['nodes'],text_key(data.get('node')))}
        elif action=='preview':result=project_store.preview(project,text_key(data.get('node')),data.get('value'))
        elif action=='apply':
            if data.get('confirm') is not True:raise ValueError('Bestätige die angezeigte CASCADE-Vorschau.')
            result=project_store.apply(project,text_key(data.get('preview')))
        elif action=='undo':result=project_store.undo(project,text_key(data.get('transaction')))
        elif action=='refresh':result=project_store.refresh(project)
        elif action=='review':result=project_store.review(project,text_key(data.get('node')))
        else:return jsonify(error='Unbekannte Projektaktion.'),404
        return jsonify(result)
    except (ValueError,KeyError) as e:return jsonify(error=str(e)),400

def text_key(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-zA-Z0-9_\-]{1,64}',value):raise ValueError('Ungültiger Schlüssel.')
    return value

def info_payload():
    return {'name':'Neral / Helios','parameters':report['trainable'],'layers':cfg.n_layers,'d_model':cfg.d_model,'heads':cfg.n_heads,'ffn_hidden':cfg.ffn_hidden,'vocab_size':cfg.vocab_size,'max_seq_len':cfg.max_seq_len,'tokenizer':'ByteTokenizer-260','device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,'checkpoint_loaded':checkpoint_loaded,'checkpoint_sha256':checkpoint_sha256,'checkpoint_meta':checkpoint_meta,'active_checkpoint':active_checkpoint,'checkpoints':list(checkpoint_paths()),'experimental':True,'runtime_status':'ready' if checkpoint_loaded else 'unavailable','runtime_message':runtime_error,'inference_only':True,'security':{'localhost_only':True,'input_limit_bytes':MAX_PROMPT_BYTES,'generation_limit_tokens':MAX_NEW_TOKENS,'rate_limit_per_minute':RATE_LIMIT,'safe_checkpoint_loading':True,'checkpoint_integrity_sha256':True,'secret_redaction':True,'prompt_guard':True,'host_origin_validation':True},'tools':['solve','code_check']}

@app.before_request
def local_and_rate_guard():
    if request.remote_addr not in {'127.0.0.1','::1'}:return jsonify(error='Nur lokale Verbindungen erlaubt.'),403
    if request.host not in ALLOWED_HOSTS:return jsonify(error='Ungültiger Host-Header.'),403
    if request.headers.get('Sec-Fetch-Site')=='cross-site':return jsonify(error='Cross-Site-Anfrage blockiert.'),403
    origin=request.headers.get('Origin')
    if origin and origin not in ALLOWED_ORIGINS:return jsonify(error='Ungültiger Origin-Header.'),403
    if request.path.startswith('/api/') and request.path!='/api/cancel':
        now=time.monotonic();key=request.remote_addr
        with _hit_lock:
            q=_hits[key]
            while q and now-q[0]>RATE_WINDOW_S:q.popleft()
            if len(q)>=RATE_LIMIT:return jsonify(error='Zu viele Anfragen. Bitte kurz warten.'),429
            q.append(now)

@app.after_request
def security_headers(resp):
    resp.headers.update({'X-Content-Type-Options':'nosniff','X-Frame-Options':'DENY','Referrer-Policy':'no-referrer','Permissions-Policy':'camera=(), microphone=(), geolocation=()','Cache-Control':'no-store','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"})
    return resp
@app.errorhandler(HTTPException)
def http_error(e):return jsonify(error=e.description),e.code
def body():
    if not request.is_json:return None,(jsonify(error='Nur JSON-Anfragen erlaubt.'),415)
    data=request.get_json(silent=True)
    if not isinstance(data,dict):return None,(jsonify(error='Ein JSON-Objekt wird erwartet.'),400)
    return data,None
@app.get('/')
def index():return render_template('index.html')
@app.get('/api/info')
def info():return jsonify(info_payload())
@app.get('/api/metrics')
def metrics():
    path=BASE/'reports/comparison.json'
    return jsonify(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'available':False})
@app.post('/api/model')
def select_model():
    data,error=body()
    if error:return error
    if not isinstance(data.get('checkpoint'),str):return jsonify(error='Checkpoint fehlt.'),400
    if not _generation_lock.acquire(blocking=False):return jsonify(error='Modell ist gerade beschäftigt.'),409
    try:load_runtime(data['checkpoint']);return jsonify(info_payload())
    except Exception:
        app.logger.exception('Checkpoint-Wechsel abgelehnt')
        return jsonify(error='Checkpoint konnte nicht sicher geladen werden. Der bisherige Zustand bleibt erhalten.'),400
    finally:_generation_lock.release()
@app.post('/api/cancel')
def cancel():
    data,error=body()
    if error:return error
    job=data.get('request_id')
    if not isinstance(job,str):return jsonify(error='Ungültige Anfrage-ID.'),400
    with _job_lock:
        event=_jobs.get(job)
        if event:event.set()
    return jsonify(cancelled=event is not None)

@torch.inference_mode()
def generate_tokens(ids,max_new,temperature,top_k,event):
    x=torch.tensor([ids],device=device,dtype=torch.long);output=[]
    for _ in range(max_new):
        if event.is_set():break
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda' and torch.cuda.is_bf16_supported()):
            logits,_=model(x);logits=logits[:,-1,:].float()
        if temperature==0:next_id=logits.argmax(-1,keepdim=True)
        else:
            logits/=temperature
            if top_k:
                threshold=torch.topk(logits,top_k).values[:,-1,None];logits=logits.masked_fill(logits<threshold,float('-inf'))
            next_id=torch.multinomial(torch.softmax(logits,dim=-1),1)
        token=int(next_id.item());output.append(token);x=torch.cat([x,next_id],dim=1)
        if token==tokenizer.eos_id:break
    return output

@app.post('/api/generate')
def generate():
    data,error=body()
    if error:return error
    prompt=data.get('prompt')
    if not isinstance(prompt,str) or not prompt.strip():return jsonify(error='Bitte gib einen Text ein.'),400
    try:prompt_bytes=len(prompt.encode('utf-8'))
    except UnicodeEncodeError:return jsonify(error='Ungültiger Unicode-Text.'),400
    if prompt_bytes>MAX_PROMPT_BYTES:return jsonify(error='Prompt überschreitet 4096 UTF-8-Bytes.'),413
    mode=data.get('mode','chat')
    if not isinstance(mode,str) or mode not in {'chat','solve','code'}:return jsonify(error='Unbekannter Modus.'),400
    params={'max_new_tokens':data.get('max_new_tokens',192),'temperature':data.get('temperature',0.0),'top_k':data.get('top_k',40)}
    if type(params['max_new_tokens']) is not int or not 1<=params['max_new_tokens']<=MAX_NEW_TOKENS:return jsonify(error='Antwortlänge muss zwischen 1 und 512 Byte-Token liegen.'),400
    if type(params['top_k']) is not int or not 0<=params['top_k']<=260:return jsonify(error='Top-k ist ungültig.'),400
    if type(params['temperature']) not in (int,float) or not math.isfinite(params['temperature']) or not 0<=params['temperature']<=1.5:return jsonify(error='Temperatur muss zwischen 0 und 1,5 liegen.'),400
    history=data.get('history',[])
    if not isinstance(history,list) or len(history)>12:return jsonify(error='Maximal zwölf vorherige Nachrichten erlaubt.'),400
    for h in history:
        if not isinstance(h,dict) or not isinstance(h.get('role'),str) or h.get('role') not in {'user','assistant'} or not isinstance(h.get('content'),str):return jsonify(error='Ungültiger Chatverlauf.'),400
    try:history_bytes=sum(len(h['content'].encode()) for h in history)
    except UnicodeEncodeError:return jsonify(error='Ungültiger Unicode-Text im Verlauf.'),400
    if history_bytes+prompt_bytes>MAX_PROMPT_BYTES:return jsonify(error='Chatkontext zu lang. Beginne einen neuen Chat.'),413
    for text in [h['content'] for h in history if h['role']=='user']+[prompt]:
        guard=check_prompt(text)
        if not guard.allowed:return jsonify(mode='guarded-response',text=guard.response,security_category=guard.category,model_generated=False)
    if mode=='code':
        try:
            result=check_code(prompt)
            text=('Python-Syntax gültig.' if result['syntax_ok'] else f"Syntaxfehler in Zeile {result['line']}: {result['message']}")
            text+='\n'+'\n'.join(f"Zeile {f['line']}: {f['message']}" for f in result['findings'])
            text+='\nDer Code wurde nicht ausgeführt. Syntaxprüfung beweist keine funktionale Korrektheit.'
            return jsonify(mode='code-tool',text=text,code_check=result,model_generated=False)
        except ValueError as e:return jsonify(error=str(e)),400
    if mode=='solve':
        try:
            result=solve(prompt)
            return jsonify(mode='solve-tool',text='\n'.join(result['steps']+[f"Ergebnis: {result['result']}"]),solve=result,model_generated=False)
        except ValueError as e:return jsonify(error=str(e)),400
    if mode=='chat' and re.fullmatch(r'[0-9x\s.+*/()%=\-^]+',prompt.strip()) and re.search(r'\d',prompt):
        try:
            result=solve(prompt)
            return jsonify(mode='solve-tool',text='\n'.join(result['steps']+[f"Ergebnis: {result['result']}"]),solve=result,model_generated=False,routing='deterministic-arithmetic-only')
        except ValueError:pass
    if not checkpoint_loaded:return jsonify(error='Kein eigener Checkpoint installiert. SOLVE, Codeprüfung und Projekte funktionieren lokal; Modellgewichte laut README bereitstellen.'),503
    formatted=''.join(('User' if h['role']=='user' else 'Assistant')+': '+h['content'].strip()+'\n' for h in history)+'User: '+prompt.strip()+'\nAssistant:'
    ids=tokenizer.encode(formatted,bos=True)
    if len(ids)+params['max_new_tokens']>cfg.max_seq_len:return jsonify(error='Kontext und Antwort überschreiten gemeinsam das Modelllimit.'),413
    job=data.get('request_id',secrets.token_hex(16))
    if not isinstance(job,str) or not re.fullmatch(r'[a-zA-Z0-9-]{16,64}',job):return jsonify(error='Ungültige Anfrage-ID.'),400
    if not _generation_lock.acquire(blocking=False):return jsonify(error='Eine Antwort wird bereits berechnet.'),429
    event=threading.Event()
    with _job_lock:_jobs[job]=event
    try:
        if device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();continuation=generate_tokens(ids,params['max_new_tokens'],params['temperature'],params['top_k'],event)
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;text,redacted=sanitize_output(tokenizer.decode(continuation))
        return jsonify(mode='trained-generation',model_generated=True,text=text,generated_tokens=len(continuation),seconds=round(elapsed,4),tokens_per_second=round(len(continuation)/max(elapsed,1e-9),2),output_secret_redacted=redacted,cancelled=event.is_set(),active_checkpoint=active_checkpoint,request_id=job)
    except Exception:
        app.logger.exception('Lokale Inferenz fehlgeschlagen')
        return jsonify(error='Die lokale Berechnung ist fehlgeschlagen. Prüfe das Serverprotokoll.'),500
    finally:
        with _job_lock:_jobs.pop(job,None)
        _generation_lock.release()
if __name__=='__main__':
    from waitress import serve
    print(f'Neral bereit: {report["trainable"]:,} Parameter, {device}, {active_checkpoint}',flush=True)
    serve(app,host='127.0.0.1',port=APP_PORT,threads=4)
