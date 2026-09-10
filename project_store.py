import storage_codec
"""Local, transactional project graph. No network or model-generated actions."""
import ast,copy,json,re,sqlite3,uuid
from datetime import datetime,timezone
from pathlib import Path
from solve import solve

def now():return datetime.now(timezone.utc).isoformat()
def uid():return uuid.uuid4().hex
def text(v,limit=1000):
    if not isinstance(v,str) or not v.strip() or len(v)>limit:raise ValueError('Text fehlt oder ist zu lang.')
    try:v.encode('utf8')
    except UnicodeEncodeError:raise ValueError('Ungültiger Unicode-Text.')
    return v.strip()
def ripple(nodes,start):
    if start not in nodes:raise ValueError('Unbekannte Annahme.')
    paths={start:[start]};queue=[start]
    while queue:
        parent=queue.pop(0)
        for key,n in nodes.items():
            if parent in n['depends_on'] and key not in paths:paths[key]=paths[parent]+[key];queue.append(key)
    return [{'id':k,'title':nodes[k]['title'],'path':p,'dependencies':nodes[k]['depends_on']} for k,p in paths.items() if k!=start]
def render(nodes):
    pending=set(nodes);done=set()
    while pending:
        ready=sorted(k for k in pending if set(nodes[k]['depends_on'])<=done)
        if not ready:raise ValueError('Abhängigkeiten enthalten einen Zyklus oder fehlen.')
        for key in ready:
            n=nodes[key]
            if n['kind']=='formula':
                formula=n['value']
                for dep in sorted(n['depends_on'],key=len,reverse=True):
                    number=nodes[dep]['result']
                    if not re.fullmatch(r'-?\d+(?:/\d+|\.\d+)?',number):raise ValueError('Formeln benötigen numerische Abhängigkeiten.')
                    formula=re.sub(r'\b'+re.escape(dep)+r'\b','('+number+')',formula)
                n['result']=solve(formula)['result']
            elif n['kind']=='answer':
                n['result']=re.sub(r'\{\{([a-z][a-z0-9_]*)\}\}',lambda m:nodes[m[1]]['result'],n['value'])
            else:n['result']=n['value']
            if len(n['result'])>4000:raise ValueError('Berechnete Antwort ist zu lang.')
            done.add(key);pending.remove(key)

def version_live(state,previous,reason,affected=None):
    stamp=now()
    for key,n in state['nodes'].items():
        if not n['live'] or (affected is not None and key not in affected):continue
        prior=previous.get(key,{});n['versions']=copy.deepcopy(prior.get('versions',n.get('versions',[])))
        n['last_checked']=stamp
        if not n['versions'] or n['versions'][-1]['text']!=n['result']:
            n['versions'].append({'version':len(n['versions'])+1,'text':n['result'],'previous':prior.get('result'),'created_at':stamp,'reason':reason})

class ProjectStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY, state TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS previews(id TEXT PRIMARY KEY, project TEXT, revision INTEGER, after_state TEXT, impacts TEXT, used INTEGER DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS transactions(id TEXT PRIMARY KEY, project TEXT, revision INTEGER, before_state TEXT, undone INTEGER DEFAULT 0)')
    def connect(self):return sqlite3.connect(self.path,timeout=10)
    def _load(self,db,project):
        row=db.execute('SELECT state FROM projects WHERE id=?',(project,)).fetchone()
        if not row:raise ValueError('Projekt nicht gefunden.')
        return storage_codec.loads(row[0])
    def _save(self,db,state):db.execute('UPDATE projects SET state=? WHERE id=?',(storage_codec.dumps(state,ensure_ascii=True),state['id']))
    def list(self):
        with self.connect() as db:return [{'id':s['id'],'name':s['name'],'revision':s['revision']} for row in db.execute('SELECT state FROM projects ORDER BY rowid') for s in [storage_codec.loads(row[0])]]
    def get(self,project):
        with self.connect() as db:return self._load(db,project)
    def create(self,name):
        name=text(name,80)
        state={'id':uid(),'name':name,'revision':0,'nodes':{},'updated_at':now(),'last_transaction':None}
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT count(*) FROM projects').fetchone()[0]>=30:raise ValueError('Maximal 30 Projekte.')
            db.execute('INSERT INTO projects VALUES (?,?)',(state['id'],storage_codec.dumps(state)))
        return state
    def add(self,project,data):
        key=text(data.get('id'),32)
        if not re.fullmatch(r'[a-z][a-z0-9_]*',key):raise ValueError('Schlüssel: Kleinbuchstaben, Ziffern und Unterstriche.')
        kind=data.get('kind')
        if kind not in ['fact','formula','answer','task']:raise ValueError('Unbekannter Eintragstyp.')
        value=text(data.get('value'),600);title=text(data.get('title'),80)
        deps=data.get('depends_on',[])
        if not isinstance(deps,list) or len(deps)>30 or any(not isinstance(d,str) for d in deps):raise ValueError('Ungültige Abhängigkeiten.')
        if kind=='formula':
            if len(value)>180:raise ValueError('Formel zu lang.')
            try:tree=ast.parse(value,mode='eval')
            except (SyntaxError,RecursionError):raise ValueError('Ungültige Formel.')
            deps=sorted(set(deps)|{n.id for n in ast.walk(tree) if isinstance(n,ast.Name)})
        if kind=='answer':deps=sorted(set(deps)|set(re.findall(r'\{\{([a-z][a-z0-9_]*)\}\}',value)))
        live=data.get('live',False)
        if type(live) is not bool or (live and kind not in ['answer','formula']):raise ValueError('Live ist für Antworten und Formeln verfügbar.')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');s=self._load(db,project)
            if key in s['nodes'] or len(s['nodes'])>=100:raise ValueError('Schlüssel existiert bereits oder Projekt ist voll.')
            if any(d not in s['nodes'] for d in deps):raise ValueError('Abhängigkeit existiert nicht.')
            s['nodes'][key]={'id':key,'title':title,'kind':kind,'value':value,'result':value,'depends_on':sorted(set(deps)),'live':live,'versions':[],'stale':False}
            render(s['nodes']);version_live(s,{},'Eintrag erstellt',affected={key});s['revision']+=1;s['updated_at']=now();s['last_transaction']=None;self._save(db,s)
        return s
    def preview(self,project,key,value):
        value=text(value,600)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');s=self._load(db,project)
            if key not in s['nodes'] or s['nodes'][key]['kind'] not in ['fact','task']:raise ValueError('Änderungen starten bei einer Annahme oder Aufgabe.')
            if value==s['nodes'][key]['value']:raise ValueError('Der Wert ist unverändert.')
            after=copy.deepcopy(s);after['nodes'][key]['value']=value;render(after['nodes']);paths=ripple(s['nodes'],key);affected={key}|{r['id'] for r in paths}
            for k in affected:
                if k!=key and after['nodes'][k]['kind'] in ['task','fact']:after['nodes'][k]['stale']=True
            impacts=[{'id':k,'title':s['nodes'][k]['title'],'before':s['nodes'][k]['result'],'after':after['nodes'][k]['result'],'needs_review':after['nodes'][k]['stale']} for k in s['nodes'] if k in affected]
            preview_id=uid();db.execute('INSERT INTO previews(id,project,revision,after_state,impacts) VALUES(?,?,?,?,?)',(preview_id,project,s['revision'],storage_codec.dumps(after),storage_codec.dumps(impacts)))
        return {'id':preview_id,'revision':s['revision'],'impacts':impacts,'paths':paths,'external_actions':False,'requires_confirmation':True}
    def apply(self,project,preview_id):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');s=self._load(db,project)
            p=db.execute('SELECT revision,after_state,impacts,used FROM previews WHERE id=? AND project=?',(preview_id,project)).fetchone()
            if not p or p[3] or p[0]!=s['revision']:raise ValueError('Vorschau ist veraltet oder bereits angewendet. Erstelle eine neue Vorschau.')
            after=storage_codec.loads(p[1]);affected={i['id'] for i in storage_codec.loads(p[2])};version_live(after,s['nodes'],'CASCADE angewendet',affected)
            tx=uid();after['revision']=s['revision']+1;after['updated_at']=now();after['last_transaction']=tx
            db.execute('INSERT INTO transactions(id,project,revision,before_state) VALUES(?,?,?,?)',(tx,project,after['revision'],storage_codec.dumps(s)))
            db.execute('UPDATE previews SET used=1 WHERE id=?',(preview_id,));self._save(db,after)
        return after
    def undo(self,project,transaction):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');s=self._load(db,project)
            row=db.execute('SELECT revision,before_state,undone FROM transactions WHERE id=? AND project=?',(transaction,project)).fetchone()
            if not row or row[2] or row[0]!=s['revision']:raise ValueError('Undo nur direkt nach der letzten CASCADE-Änderung möglich.')
            restored=storage_codec.loads(row[1]);version_live(restored,s['nodes'],'CASCADE rückgängig gemacht');restored['revision']=s['revision']+1;restored['updated_at']=now();restored['last_transaction']=None
            db.execute('UPDATE transactions SET undone=1 WHERE id=?',(transaction,));self._save(db,restored)
        return restored
    def refresh(self,project):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');s=self._load(db,project);before=copy.deepcopy(s['nodes']);render(s['nodes']);version_live(s,before,'Lokale Quellen erneut geprüft');s['revision']+=1;s['updated_at']=now();s['last_transaction']=None;self._save(db,s)
        return s
    def review(self,project,key):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');s=self._load(db,project)
            if key not in s['nodes'] or not s['nodes'][key]['stale']:raise ValueError('Kein offener Prüfhinweis für diesen Eintrag.')
            s['nodes'][key]['stale']=False;s['revision']+=1;s['updated_at']=now();s['last_transaction']=None;self._save(db,s)
        return s
