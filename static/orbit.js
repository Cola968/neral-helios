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
function taskCard(t){
  const root=el('article',undefined,'task');
  root.append(el('div',t.kind+' · '+t.status,'pill'),el('h3',t.title),el('p',t.goal));
  const list=el('ul');
  for(const step of t.plan){list.append(el('li',step.description));}
  root.append(list);
  const row=el('div',undefined,'task-controls');
  async function act(action){
    try{await call('/api/orbit/tasks/'+encodeURIComponent(t.id)+'/transition',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action,revision:t.revision})});
      status(action==='approve'?'Freigabe gespeichert. Kein Worker wurde gestartet.':action==='archive'?'Abgebrochener Auftrag archiviert; Verlauf bleibt lokal gespeichert.':'Auftrag abgebrochen.');
      await reload();
    }catch(err){status(err.message);await reload();}
  }
  if(t.status==='DRAFT'){
    const approve=el('button','Freigabe notieren');approve.addEventListener('click',()=>act('approve'));row.append(approve);
  }
  if(t.status==='DRAFT'||t.status==='APPROVED_NO_EXECUTOR'){
    const cancel=el('button','Abbrechen','danger');cancel.addEventListener('click',()=>act('cancel'));row.append(cancel);
  }
  if(t.status==='CANCELLED'){
    const archive=el('button','Archivieren');archive.addEventListener('click',()=>act('archive'));row.append(archive);
  }
  root.append(row);
  if(t.status==='APPROVED_NO_EXECUTOR')root.append(el('p','Freigegeben, aber nicht ausgeführt. Noch kein Cloud-Worker aktiv.'));
  return root;
}
async function reload(){
  const list=$('tasks');
  try{
    const data=await call('/api/orbit/tasks?view='+(showingArchived?'archived':'active'));
    $('showArchived').textContent=showingArchived?'Aktive Aufträge':'Archiv anzeigen';
    $('showArchived').setAttribute('aria-pressed',String(showingArchived));
    $('taskListHeading').textContent=showingArchived?'Archivierte Aufträge':'Deine Aufträge';
    list.replaceChildren();
    if(!Array.isArray(data.tasks)||!data.tasks.length)list.append(el('p',showingArchived?'Keine archivierten Aufträge.':'Noch keine Aufträge erstellt.','empty'));
    else for(const task of data.tasks)list.append(taskCard(task));
  }catch(err){list.replaceChildren(el('p',err.message+' Aktiviere den lokalen Prototyp mit HELIOS_ORBIT_ENABLED=1.','empty'));}
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
reload();
