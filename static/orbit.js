'use strict';
const $=id=>document.getElementById(id);
async function call(path,options={}){
  let response;
  try{
    response=await fetch(path,{credentials:'same-origin',...options});
  }catch{throw Error('Verbindung fehlgeschlagen. Bitte NERAL lokal starten.');}
  let data;try{data=await response.json();}catch{throw Error('Ungültiges Antwortformat des Servers.');}
  if(!response.ok)throw Error(typeof data.error==='string'?data.error:'Anfrage fehlgeschlagen.');
  return data;
}
function el(tag,text,cls){
  const e=document.createElement(tag);
  if(typeof text==='string')e.textContent=text;
  if(cls)e.className=cls;
  return e;
}
function status(text){$('status').textContent=text;}
let showingArchived=false;
let archiveOffset=0;
function taskCard(t){
  const root=el('article',undefined,'task');
  root.append(el('div',t.kind+' · '+t.status,'pill'),el('h3',t.title),el('p',t.goal));
  const list=el('ul');
  const stageLabels={NOT_STARTED:'Nicht begonnen',APPROVED:'Freigegeben',
    IN_PROGRESS:'Läuft',COMPLETED:'Abgeschlossen',
    NEEDS_REVIEW:'Unabhängige Prüfung offen',FAILED:'Fehlgeschlagen'};
  for(const step of t.plan){
    const label=stageLabels[step.status]||'Unbekannt';
    const item=el('li');
    item.append(el('span',step.description),el('small',' · '+label,'stage-status'));
    list.append(item);
  }
  root.append(list);
  const row=el('div',undefined,'task-controls');
  async function act(action){
    try{await call('/api/orbit/tasks/'+encodeURIComponent(t.id)+'/transition',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action,revision:t.revision})});
      status(action==='approve'?'Freigabe gespeichert. Noch keine Recherche gestartet.':action==='archive'?'Auftrag archiviert; Verlauf bleibt lokal gespeichert.':action==='review'?'Bericht als gelesen markiert.':'Auftrag abgebrochen.');
      await reload();
    }catch(err){status(err.message);await reload();}
  }
  if(t.status==='DRAFT'){
    const approve=el('button','Freigabe notieren');approve.addEventListener('click',()=>act('approve'));row.append(approve);
  }
  if(t.status==='DRAFT'||t.status==='APPROVED_NO_EXECUTOR'){
    const cancel=el('button','Abbrechen','danger');cancel.addEventListener('click',()=>act('cancel'));row.append(cancel);
  }
  if(t.status==='APPROVED_NO_EXECUTOR'&&t.kind==='research'){
    const run=el('button','Wikipedia-Recherche jetzt starten');
    run.addEventListener('click',async ()=>{
      if(!window.confirm('Öffentliche Recherchefrage einmalig an die Wikipedia senden? Noch kein KI-Modell oder Hintergrundagent.'))return;
      run.disabled=true;
      try{
        const data=await call('/api/orbit/tasks/'+encodeURIComponent(t.id)+'/run-research',{
          method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({revision:t.revision,confirm:true})
        });
        status('Öffentliche Quellen abgerufen und für deine Prüfung gespeichert.');
        await reload();
      }catch(err){status(err.message);await reload();}
      finally{run.disabled=false;}
    });
    row.append(run);
  }
  if(t.status==='REVIEW_READY'){
    const reviewed=el('button','Bericht gelesen');reviewed.addEventListener('click',()=>act('review'));row.append(reviewed);
  }
  if(t.status==='CANCELLED'||t.status==='REVIEWED'||t.status==='RESEARCH_FAILED'){
    const archive=el('button','Archivieren');archive.addEventListener('click',()=>act('archive'));row.append(archive);
  }
  root.append(row);
  if(t.status==='APPROVED_NO_EXECUTOR')root.append(el('p','Freigegeben, aber noch nicht ausgeführt. Eine Recherche startet erst durch den gesonderten Klick.'));
  if(t.status==='RESEARCH_RUNNING')root.append(el('p','Einmalige Recherche läuft; keine Hintergrundüberwachung.'));
  if(t.error)root.append(el('p',t.error,'task-error'));
  if(t.report&&Array.isArray(t.report.sources)){
    const header=el('p','Wikipedia-Auszüge — bitte selbst prüfen; keine KI-Zusammenfassung.','report-heading');
    root.append(header);
    const references=el('ol',undefined,'source-list');
    for(const source of t.report.sources){
      const item=el('li');
      const title=el('a',source.title);
      if(/^https:\/\/de\.wikipedia\.org\/\?curid=[1-9]\d{0,15}$/.test(source.url)){
        title.href=source.url;title.target='_blank';title.rel='noopener noreferrer';
      }
      item.append(title,el('p',source.excerpt));
      references.append(item);
    }
    root.append(references);
    const download=el('a','Quellenbericht als JSON speichern','report-download');
    download.href='/api/orbit/tasks/'+encodeURIComponent(t.id)+'/report';
    download.download='neral-orbit-'+t.id+'.json';
    root.append(download);
    root.append(el('p',t.report.limitations||'Quellen sind nicht unabhängig überprüft.','report-limitations'));
  }
  return root;
}
async function reload(){
  archiveOffset=0;
  const list=$('tasks');
  try{
    const data=await call('/api/orbit/tasks?view='+(showingArchived?'archived':'active'));
    $('archiveMore').hidden=!showingArchived||!data.has_more;
    $('showArchived').textContent=showingArchived?'Aktive Aufträge':'Archiv anzeigen';
    $('showArchived').setAttribute('aria-pressed',String(showingArchived));
    $('taskListHeading').textContent=showingArchived?'Archivierte Aufträge':'Deine Aufträge';
    list.replaceChildren();
    if(!Array.isArray(data.tasks)||!data.tasks.length)list.append(el('p',showingArchived?'Keine archivierten Aufträge.':'Noch keine Aufträge erstellt.','empty'));
    else for(const task of data.tasks)list.append(taskCard(task));
  }catch(err){
    $('archiveMore').hidden=true;
    list.replaceChildren(el('p',err.message+' Aktiviere den lokalen Prototyp mit HELIOS_ORBIT_ENABLED=1.','empty'));
  }
}
$('taskForm').addEventListener('submit',async e=>{
  e.preventDefault();
  $('create').disabled=true;
  try{
    const form=new FormData(e.currentTarget);
    await call('/api/orbit/tasks',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({kind:form.get('kind'),title:form.get('title'),goal:form.get('goal')})});
    $('taskForm').reset();status('Auftrag gespeichert – noch nicht gestartet.');await reload();
  }catch(err){status(err.message);}
  finally{$('create').disabled=false;}
});
$('refresh').addEventListener('click',reload);
$('showArchived').addEventListener('click',()=>{showingArchived=!showingArchived;reload();});
$('archiveMore').addEventListener('click',async ()=>{
  if(!showingArchived)return;
  const button=$('archiveMore'),next=archiveOffset+40;
  if(next>100000){status('Archivgrenze erreicht.');return;}
  button.disabled=true;
  try{
    const data=await call('/api/orbit/tasks?view=archived&offset='+next);
    if(!showingArchived)return; // Do not display stale results after switching views.
    if(!Array.isArray(data.tasks))throw Error('Ungültige Archivantwort.');
    for(const task of data.tasks)$('tasks').append(taskCard(task));
    archiveOffset=next;
    button.hidden=!data.has_more;
  }catch(err){status(err.message);}
  finally{button.disabled=false;}
});
reload();
