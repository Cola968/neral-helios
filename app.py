"""Local Neral/Helios browser test: own unchanged 20M architecture."""
from __future__ import annotations
import hashlib,io,json,math,os,re,secrets,threading,time
from collections import defaultdict,deque
from pathlib import Path
import torch
from flask import Flask,jsonify,render_template,request,send_from_directory,Response
from werkzeug.exceptions import HTTPException
from config import NeralConfig
from model import Neral20M
from tokenizer import ByteTokenizer
from security_guard import check_prompt,sanitize_output
from solve import solve
from chat_math import tool_answer
from code_check import check_code
from architectures import expected_parameters
from project_store import ProjectStore,ripple
from chat_store import ChatStore,Conflict
from stream_runtime import response_stream
from model_catalog import catalog
import arctus_runtime
from neral_orbit import OrbitStore, OrbitError, OrbitConflict
from neral_orbit_research import run_public_research, validate_public_query, ResearchError
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
app.config.update(MAX_CONTENT_LENGTH=256*1024,JSON_SORT_KEYS=False)
STATE=Path(os.environ.get('HELIOS_STATE_ROOT',str(BASE/'local_state'))).resolve();STATE.mkdir(parents=True,exist_ok=True)
SESSION_TOKEN=secrets.token_urlsafe(32)
(STATE/'gateway-token').write_text(SESSION_TOKEN,encoding='ascii')
try:os.chmod(STATE/'gateway-token',0o600)
except OSError:pass
project_store=ProjectStore(STATE/'projects.sqlite3')
chat_store=ChatStore(STATE/'chats.sqlite3')
# Orbit stores local task drafts; research execution requires separate explicit approval+launch.
orbit_store=OrbitStore(STATE/'orbit.sqlite3')
ORBIT_ENABLED=os.environ.get('HELIOS_ORBIT_ENABLED')=='1'
if ORBIT_ENABLED:
    # Crash recovery only changes old local RUNNING records to FAILED.
    # It never retries a network request or schedules a worker.
    orbit_store.reconcile_stale_research(min_age_seconds=60)
_hits=defaultdict(deque);_hit_lock=threading.Lock();_generation_lock=threading.Lock();_jobs={};_job_lock=threading.Lock()
torch.set_num_threads(2);torch.manual_seed(968);torch.set_float32_matmul_precision('high')
cfg=NeralConfig();model=Neral20M(cfg);tokenizer=ByteTokenizer();report=model.parameter_report()
assert report['trainable']==EXPECTED_PARAMS
PREPARATION_MODE=os.environ.get('HELIOS_PREPARATION_MODE','0')=='1'
device_name='cpu' if PREPARATION_MODE else os.environ.get('HELIOS_DEVICE','cpu')
if device_name!='cpu':raise ValueError('This Helios app release is CPU-only.')
device=torch.device(device_name)
runtime_error=None
checkpoint_loaded=False;checkpoint_meta={};checkpoint_sha256=None;active_checkpoint='original'
arctus=None
def runtime_identity():
    return arctus.identity if arctus is not None else ('neral-byte:'+checkpoint_sha256 if checkpoint_loaded and checkpoint_sha256 else None)
def model_family():return arctus_runtime.FAMILY if arctus is not None else 'neral-byte'
def imported_models():
    try:return catalog(ASSETS)
    except (ValueError,OSError,KeyError):return {}
def checkpoint_paths():
    paths={'original':(ASSETS/'checkpoints/neral20m.pt',ASSETS/'checkpoints/neral20m.sha256')}
    if (ASSETS/'export/neral20m.pt').exists():paths['pilot_01']=(ASSETS/'export/neral20m.pt',ASSETS/'export/neral20m.sha256')
    if (ASSETS/'export/pilot_02/neral20m.pt').exists():paths['pilot_02']=(ASSETS/'export/pilot_02/neral20m.pt',ASSETS/'export/pilot_02/neral20m.sha256')
    if (ASSETS/'export/neral40m/neral40m.pt').exists():paths['neral40m']=(ASSETS/'export/neral40m/neral40m.pt',ASSETS/'export/neral40m/neral40m.sha256')
    if (ASSETS/'export/neral50m/neral50m.pt').exists():paths['neral50m']=(ASSETS/'export/neral50m/neral50m.pt',ASSETS/'export/neral50m/neral50m.sha256')
    i1=ASSETS/'export/neral50m_i1/neral50m.pt';digest=i1.with_suffix('.sha256')
    if i1.is_file() and digest.is_file():paths['neral50m_i1']=(i1,digest)
    paths.update({key:row["paths"] for key,row in imported_models().items()})
    return paths
def load_runtime(name):
    global checkpoint_loaded,checkpoint_meta,checkpoint_sha256,active_checkpoint,model,cfg,report,runtime_error,arctus
    if PREPARATION_MODE:raise ValueError('Vorbereitungsmodus: Modellladen ist deaktiviert.')
    if name=='none':
        arctus=None;model=None;checkpoint_loaded=False;checkpoint_meta={};checkpoint_sha256=None;active_checkpoint='none'
        runtime_error='Kein natives Modell geladen. Lokale Werkzeuge sind bereit.'
        return
    if name==arctus_runtime.MODEL_ID:
        candidate=arctus_runtime.load()
        # Commit only after the entire candidate has been verified and loaded.
        arctus=candidate;model=None;checkpoint_loaded=True;checkpoint_meta={};checkpoint_sha256=candidate.sha256
        active_checkpoint=name;runtime_error=None
        return
    if name not in checkpoint_paths():raise ValueError('Unbekannter Checkpoint.')
    path,hash_path=checkpoint_paths()[name]
    if not path.is_file() or not hash_path.is_file():raise ValueError('Checkpoint oder SHA-256-Datei fehlt.')
    if path.stat().st_size>200*1024*1024:raise ValueError('Inferenz-Checkpoint zu groß.')
    raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest();hash_parts=hash_path.read_text(encoding='ascii').split();expected=hash_parts[0] if hash_parts else ''
    if not re.fullmatch('[a-fA-F0-9]{64}',expected) or not secrets.compare_digest(actual,expected.lower()):raise ValueError('Checkpoint-Integrität fehlgeschlagen.')
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);state=payload['model']
    candidate_cfg=NeralConfig(**payload.get('config',{}));expected=expected_parameters(candidate_cfg)
    if name=='neral50m_i1' and expected!=50_012_976:raise ValueError('I1 erfordert exakt Neral-50M.')
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
    arctus=None
def initialize_runtime():
    global runtime_error
    if PREPARATION_MODE:
        runtime_error='Vorbereitungsmodus: keine Modellinferenz. SOLVE, Codeprüfung und Projekte sind bereit.'
        return
    name=os.environ.get('HELIOS_CHECKPOINT') or ('pilot_02' if (ASSETS/'export/pilot_02/neral20m.pt').is_file() else 'original')
    if name==arctus_runtime.MODEL_ID:
        try:load_runtime(name)
        except arctus_runtime.ArctusError as exc:runtime_error=str(exc)
        return
    if name not in checkpoint_paths():
        runtime_error='Der ausgewählte eigene Checkpoint ist nicht installiert.'
        return
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
    payload={'app_version':'2.0','name':'Neral / Helios','model_family':model_family(),'runtime_id':runtime_identity(),'external_inference':False,'native_arctus350_integrated':True,'arctus350_enabled':arctus_runtime.enabled(),'parameters':report['trainable'],'layers':cfg.n_layers,'d_model':cfg.d_model,'heads':cfg.n_heads,'ffn_hidden':cfg.ffn_hidden,'vocab_size':cfg.vocab_size,'max_seq_len':cfg.max_seq_len,'tokenizer':'ByteTokenizer-260','device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,'checkpoint_loaded':checkpoint_loaded,'checkpoint_sha256':checkpoint_sha256,'checkpoint_meta':checkpoint_meta,'active_checkpoint':active_checkpoint,'checkpoints':list(checkpoint_paths())+([arctus_runtime.MODEL_ID] if arctus_runtime.enabled() else [])+['none'],'model_labels':{**{k:v['label'] for k,v in imported_models().items()},'arctus350':arctus_runtime.LABEL,'none':'Modell entladen · nur Werkzeuge'},'experimental':True,'runtime_status':'ready' if checkpoint_loaded else 'unavailable','runtime_message':runtime_error,'inference_only':True,'preparation_mode':PREPARATION_MODE,'i1_export_available':'neral50m_i1' in checkpoint_paths(),'security':{'localhost_only':True,'input_limit_bytes':MAX_PROMPT_BYTES,'generation_limit_tokens':MAX_NEW_TOKENS,'rate_limit_per_minute':RATE_LIMIT,'safe_checkpoint_loading':True,'checkpoint_integrity_sha256':True,'secret_redaction':True,'prompt_guard':True,'host_origin_validation':True},'tools':['solve','code_check']}
    if arctus is not None:
        c=arctus_runtime.EXPECTED_CONFIG
        payload.update(parameters=arctus_runtime.EXPECTED_PARAMS,layers=c['n_layers'],d_model=c['d_model'],heads=c['n_heads'],ffn_hidden=c['ffn_hidden'],vocab_size=c['vocab_size'],max_seq_len=arctus_runtime.APP_CONTEXT,tokenizer='NERAL-16K-BPE',device='cpu',gpu=None)
    return payload
@app.before_request
def local_and_rate_guard():
    if request.remote_addr not in {'127.0.0.1','::1'}:return jsonify(error='Nur lokale Verbindungen erlaubt.'),403
    if request.host not in ALLOWED_HOSTS:return jsonify(error='Ungültiger Host-Header.'),403
    if request.headers.get('Sec-Fetch-Site')=='cross-site':return jsonify(error='Cross-Site-Anfrage blockiert.'),403
    origin=request.headers.get('Origin')
    if origin and origin not in ALLOWED_ORIGINS:return jsonify(error='Ungültiger Origin-Header.'),403
    if request.path.startswith('/api/'):
        supplied=request.cookies.get('neral_local','') or request.headers.get('X-Helios-Gateway','')
        if not secrets.compare_digest(supplied,SESSION_TOKEN):return jsonify(error='Sitzung abgelaufen. Bitte App neu laden.',code='session_expired'),401
    if request.path.startswith('/api/'):
        now=time.monotonic();key=(request.remote_addr,'generation' if request.path=='/api/generate' else 'control')
        with _hit_lock:
            q=_hits[key]
            while q and now-q[0]>RATE_WINDOW_S:q.popleft()
            if len(q)>=(RATE_LIMIT if key[1]=='generation' else 180):return jsonify(error='Zu viele Anfragen. Bitte kurz warten.'),429
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
def index():
    response=app.make_response(render_template('index.html'))
    response.set_cookie('neral_local',SESSION_TOKEN,httponly=True,samesite='Strict',max_age=28800,secure=False)
    return response
@app.get('/orbit')
def orbit_index():
    # The prototype page can explain opt-in even when the feature is disabled.
    response=app.make_response(render_template('orbit.html'))
    response.set_cookie('neral_local',SESSION_TOKEN,httponly=True,samesite='Strict',max_age=28800,secure=False)
    return response

@app.route('/api/orbit/tasks',methods=['GET','POST'])
def orbit_tasks():
    if not ORBIT_ENABLED:return jsonify(error='NERAL Orbit v0 ist nicht aktiviert.'),404
    if request.method=='GET':
        view=request.args.get('view','active')
        if view not in ('active','archived'):
            return jsonify(error='Unbekannte Aufgabenansicht.'),400
        return jsonify(tasks=orbit_store.list(archived=view=='archived'),
                       executor_active=False,cloud_scheduled=False)
    data,error=body()
    if error:return error
    try:
        task=orbit_store.create(kind=data.get('kind'),title=data.get('title'),goal=data.get('goal'))
        return jsonify(task=task),201
    except OrbitError as exc:return jsonify(error=str(exc)),400

@app.post('/api/orbit/tasks/<task_id>/transition')
def orbit_transition(task_id):
    if not ORBIT_ENABLED:return jsonify(error='NERAL Orbit v0 ist nicht aktiviert.'),404
    data,error=body()
    if error:return error
    try:
        updated=orbit_store.transition(task_id,data.get('action'),revision=data.get('revision'))
        return jsonify(task=updated)
    except OrbitConflict as exc:return jsonify(error=str(exc)),409
    except OrbitError as exc:return jsonify(error=str(exc)),400

@app.get('/api/orbit/tasks/<task_id>/report')
def orbit_download_report(task_id):
    # Only previously completed source evidence, no new network requests.
    # Session/host protection is the same as for other Orbit endpoints.
    if not ORBIT_ENABLED:return jsonify(error='NERAL Orbit ist nicht aktiviert.'),404
    try:
        task=orbit_store.get(task_id)
    except OrbitError as exc:return jsonify(error=str(exc)),400
    report=task.get('report')
    if task['status'] not in ('REVIEW_READY','REVIEWED','ARCHIVED') or not report:
        return jsonify(error='Kein abgeschlossener Quellenbericht vorhanden.'),409
    result=jsonify(report)
    result.headers['Content-Disposition']=f'attachment; filename="neral-orbit-{task_id}.json"'
    result.headers['Cache-Control']='no-store'
    return result

@app.post('/api/orbit/tasks/<task_id>/run-research')
def orbit_run_research(task_id):
    # One deliberate manual request. No timers, auto-jobs, remote shells,
    # arbitrary URLs, AI model calls, payments, or unattended execution.
    if not ORBIT_ENABLED:return jsonify(error='NERAL Orbit ist nicht aktiviert.'),404
    data,error=body()
    if error:return error
    if data.get('confirm') is not True:
        return jsonify(error='Ausdrückliche Bestätigung für die einmalige öffentliche Recherche fehlt.'),400
    try:
        # Check the immutable task goal BEFORE reserving execution or
        # sending a possibly private query to the public provider.
        candidate=orbit_store.get(task_id)
        validate_public_query(candidate['goal'])
        task=orbit_store.begin_research(task_id,revision=data.get('revision'))
    except OrbitConflict as exc:return jsonify(error=str(exc)),409
    except (OrbitError,ResearchError) as exc:return jsonify(error=str(exc)),400
    try:
        report=run_public_research(task['goal'])
    except ResearchError as exc:
        failed=orbit_store.finish_research(task_id,revision=task['revision'],error=str(exc)[:160])
        return jsonify(error=str(exc),task=failed),502
    except Exception:
        failed=orbit_store.finish_research(
            task_id,revision=task['revision'],
            error='Unbekannter Recherchefehler. Kein Ergebnis wurde erfunden.')
        return jsonify(error=failed['error'],task=failed),502
    result=orbit_store.finish_research(task_id,revision=task['revision'],report=report)
    return jsonify(task=result,source_review_required=True)

@app.get('/manifest.webmanifest')
def manifest():return send_from_directory(BASE/'static','manifest.webmanifest',mimetype='application/manifest+json')
@app.get('/sw.js')
def service_worker():return send_from_directory(BASE/'static','sw.js',mimetype='text/javascript')
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
    except arctus_runtime.ArctusError as exc:return jsonify(error=str(exc)),400
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
def generate_tokens(ids,max_new,temperature,top_k,event,on_tokens=None):
    if arctus is not None:return arctus.generate(ids,max_new,temperature,top_k,event,on_tokens)
    x=torch.tensor([ids],device=device,dtype=torch.long);output=[];deadline=time.monotonic()+120
    for _ in range(max_new):
        if time.monotonic()>deadline:event.set()
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
        if on_tokens and (len(output)%4==0 or token==tokenizer.eos_id):on_tokens(tokenizer.decode(output),len(output))
        if token==tokenizer.eos_id:break
    return output
@app.post('/api/generate')
def generate():
    data,error=body()
    if error:return error
    if type(data.get('stream',False)) is not bool:return jsonify(error='Ungültige Stream-Option.'),400
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
    if len(history)%2:return jsonify(error='Chatverlauf muss vollständige Nachrichtenpaare enthalten.'),400
    for index,h in enumerate(history):
        if not isinstance(h,dict) or not isinstance(h.get('role'),str) or h.get('role') not in {'user','assistant'} or not isinstance(h.get('content'),str):return jsonify(error='Ungültiger Chatverlauf.'),400
        if h['role']!=('user' if index%2==0 else 'assistant'):return jsonify(error='Ungültige Reihenfolge im Chatverlauf.'),400
    try:history_bytes=sum(len(h['content'].encode()) for h in history)
    except UnicodeEncodeError:return jsonify(error='Ungültiger Unicode-Text im Verlauf.'),400
    if history_bytes+prompt_bytes>MAX_PROMPT_BYTES:return jsonify(error='Chatkontext zu lang. Beginne einen neuen Chat.'),413
    for text in [h['content'] for h in history if h['role']=='user']+[prompt]:
        guard=check_prompt(text)
        if not guard.allowed:return jsonify(mode='guarded-response',source='local-rule:security',text=guard.response,security_category=guard.category,model_generated=False)
    if mode=='code':
        try:
            result=check_code(prompt)
            text=('Python-Syntax gültig.' if result['syntax_ok'] else f"Syntaxfehler in Zeile {result['line']}: {result['message']}")
            text+='\n'+'\n'.join(f"Zeile {f['line']}: {f['message']}" for f in result['findings'])
            text+='\nDer Code wurde nicht ausgeführt. Syntaxprüfung beweist keine funktionale Korrektheit.'
            return jsonify(mode='code-tool',source='tool:code-check',text=text,code_check=result,model_generated=False)
        except ValueError as e:return jsonify(error=str(e)),400
    if mode=='solve':
        try:
            result=solve(prompt.translate(str.maketrans({'×':'*','÷':'/','−':'-'})))
            return jsonify(mode='solve-tool',source='tool:solve',text='\n'.join(result['steps']+[f"Ergebnis: {result['result']}"]),solve=result,model_generated=False)
        except ValueError as e:return jsonify(error=str(e)),400
    if mode=='chat':
        calculated=tool_answer(prompt)
        if calculated is not None:
            return jsonify(mode='solve-tool',source='tool:'+calculated.kind,text=calculated.text,
                           solve={'result':calculated.text,'engine':'bounded-fraction-calculator','model_generated':False},
                           model_generated=False,routing='whole-request-math')
    if mode=='chat' and re.fullmatch(r'[0-9x\s.+*/()%=\-^]+',prompt.strip()) and re.search(r'\d',prompt):
        try:
            result=solve(prompt)
            return jsonify(mode='solve-tool',source='tool:solve',text='\n'.join(result['steps']+[f"Ergebnis: {result['result']}"]),solve=result,model_generated=False,routing='deterministic-arithmetic-only')
        except ValueError:pass
    if PREPARATION_MODE:return jsonify(error='Vorbereitungsmodus: Kein natives Modell aktiv. Vollständige Rechenausdrücke funktionieren hier; mehrstufige Textaufgaben bitte als einen Ausdruck mit Klammern formulieren.',code='native_unavailable',source='no-model',model_generated=False),503
    job=data.get('request_id',secrets.token_hex(16))
    if not isinstance(job,str) or not re.fullmatch(r'[a-zA-Z0-9-]{16,64}',job):return jsonify(error='Ungültige Anfrage-ID.'),400
    if not _generation_lock.acquire(blocking=False):return jsonify(error='Eine Antwort wird bereits berechnet.'),429
    # Keep runtime selection, tokenization, inference and decoding under one lock.
    # A second tab cannot switch the tokenizer between encoding and generation.
    try:
        if not checkpoint_loaded:
            _generation_lock.release()
            return jsonify(error='Kein eigenes Sprachmodell geladen. Rechenwerkzeuge, Codeprüfung und Projekte sind bereit.',code='native_unavailable',source='no-model',model_generated=False),503
        identity=runtime_identity()
        if (arctus is not None and data.get('runtime_id')!=identity) or ('runtime_id' in data and data['runtime_id']!=identity):
            _generation_lock.release()
            return jsonify(error='Das aktive Modell wurde gewechselt. Status aktualisieren und erneut senden.'),409
        if arctus is not None:ids=arctus.encode(history,prompt,params['max_new_tokens'])
        else:
            # Reject explicitly foreign context; legacy byte-only clients remain compatible.
            if any(h.get('model_family','neral-byte')!='neral-byte' or (h.get('runtime_id') is not None and h['runtime_id']!=identity) for h in history):
                _generation_lock.release()
                return jsonify(error='Der Verlauf gehört zu einem anderen Modell.'),409
            formatted=''.join(('User' if h['role']=='user' else 'Assistant')+': '+h['content'].strip()+'\n' for h in history)+'User: '+prompt.strip()+'\nAssistant:'
            ids=tokenizer.encode(formatted,bos=True)
            if len(ids)+params['max_new_tokens']>cfg.max_seq_len:
                _generation_lock.release()
                return jsonify(error='Kontext und Antwort überschreiten gemeinsam das Modelllimit.'),413
    except arctus_runtime.ArctusError as exc:
        _generation_lock.release()
        return jsonify(error=str(exc)),400
    except Exception:
        _generation_lock.release()
        raise
    event=threading.Event()
    with _job_lock:_jobs[job]=event
    def finish():
        with _job_lock:_jobs.pop(job,None)
        _generation_lock.release()
    def run(callback=None):
        start=time.perf_counter()
        args=(ids,params['max_new_tokens'],params['temperature'],params['top_k'],event)
        continuation=generate_tokens(*args,on_tokens=callback) if callback else generate_tokens(*args)
        decoded=arctus.decode(continuation) if arctus is not None else tokenizer.decode(continuation)
        elapsed=time.perf_counter()-start;text,redacted=sanitize_output(decoded)
        return dict(mode='trained-generation',source='native-model',model_family=model_family(),runtime_id=identity,token_unit='BPE-Token' if arctus is not None else 'Byte-Token',finish_reason=arctus.finish_reason if arctus is not None else None,incomplete=arctus is not None and arctus.finish_reason!='eos',model_generated=True,text=text,generated_tokens=len(continuation),seconds=round(elapsed,4),tokens_per_second=round(len(continuation)/max(elapsed,1e-9),2),output_secret_redacted=redacted,cancelled=event.is_set(),active_checkpoint=active_checkpoint,request_id=job)
    if data.get('stream'):
        response=Response(response_stream(run,event,finish),mimetype='application/x-ndjson')
        response.headers['X-Accel-Buffering']='no'
        response.call_on_close(event.set)
        return response
    try:return jsonify(run())
    except Exception:
        app.logger.exception('Lokale Inferenz fehlgeschlagen')
        return jsonify(error='Die lokale Berechnung ist fehlgeschlagen. Prüfe das Serverprotokoll.'),500
    finally:finish()
@app.get('/api/chats')
def list_chats():return jsonify(chats=chat_store.all())
@app.post('/api/chats/<key>')
def save_chat(key):
    data,error=body()
    if error:return error
    try:return jsonify(revision=chat_store.put(key,data.get('revision'),data.get('chat'),data.get('deleted',False)))
    except Conflict:return jsonify(error='Versionskonflikt. Lokale Kopie behalten und neu synchronisieren.',code='revision_conflict'),409
    except ValueError as e:return jsonify(error=str(e)),400
@app.get('/api/health')
def health():return jsonify(status='ready' if checkpoint_loaded else 'tools-only',device='cpu',busy=_generation_lock.locked(),streaming=True,sync=True)
@app.errorhandler(Exception)
def unexpected_error(error):
    incident=secrets.token_hex(6)
    app.logger.error('App incident %s: %s',incident,type(error).__name__)
    return jsonify(error='Interner Fehler. Deine lokalen Entwürfe bleiben erhalten.',incident=incident),500
if __name__=='__main__':
    from waitress import serve
    print(f'Neral bereit: {report["trainable"]:,} Parameter, {device}, {active_checkpoint}',flush=True)
    serve(app,host='127.0.0.1',port=APP_PORT,threads=4)

