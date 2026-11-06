'use strict';
/* Local model-directed workflow, no model endpoint or untrusted HTML in browser. */
(() => {
  const el = id => document.getElementById(id);
  if (!el('agentAutoForm')) return;
  let job = null, working = false, abortRequested = false;
  const controls = {
    next:el('agentAutoNext'), run:el('agentAutoRun'),pause:el('agentAutoPause'),
    resume:el('agentAutoResume'),cancel:el('agentAutoCancel')
  };
  async function post(path, body={}) {
    const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw Error(d.error||'Lokaler Agentenaufruf fehlgeschlagen.');
    return d;
  }
  function message(e) { el('agentAutoState').textContent=e.message||String(e); }
  function display(data) {
    job = data;
    const run = data.status==='running';
    controls.next.disabled=!run||working;
    controls.run.disabled=!run||working;
    controls.pause.disabled=!run;
    controls.resume.disabled=data.status!=='paused';
    controls.cancel.disabled=!['running','paused','awaiting_approval'].includes(data.status);
    el('agentAutoState').textContent=data.status+' · '+data.steps+'/'+data.max_steps+' Schritte'+
      (data.answer?' · '+data.answer:'');
    el('agentAutoEvidence').textContent=JSON.stringify(data.history,null,2);
    const pending=data.status==='awaiting_approval' && data.proposal;
    el('agentAutoApproval').hidden=!pending;
    el('agentAutoDiff').textContent=pending?(data.proposal.diff||'Neue Datei'):''; 
  }
  el('agentAutoForm').addEventListener('submit', async e=>{
    e.preventDefault();abortRequested=false;
    try{
      const data=await post('/api/agent/auto/start',{goal:el('agentAutoGoal').value});
      display(data);
    }catch(err){message(err);}
  });
  async function action(name, payload={}) {
    if (!job) return;
    try { display(await post('/api/agent/auto/'+job.id+'/'+name,payload)); }
    catch(err) { message(err); }
  }
  controls.next.addEventListener('click',()=>action('step'));
  controls.pause.addEventListener('click',()=>{abortRequested=true;action('pause');});
  controls.resume.addEventListener('click',()=>{abortRequested=false;action('resume');});
  controls.cancel.addEventListener('click',()=>{abortRequested=true;action('cancel');});
  controls.run.addEventListener('click',async()=>{
    if (!job||working) return;
    working=true; abortRequested=false;
    controls.run.disabled=true; controls.next.disabled=true;
    try {
      // Each request is one model action; UI remains responsive between actions.
      for(let i=0;i<12&&!abortRequested&&job?.status==='running';i++){
        display(await post('/api/agent/auto/'+job.id+'/step'));
      }
    } catch(err) {message(err);}
    finally {working=false;if(job)display(job);}
  });
  el('agentAutoApprove').addEventListener('click',async()=>{
    if(!job?.proposal?.token || !window.confirm('Den angezeigten Diff auf deinem PC schreiben?'))return;
    await action('approve',{token:job.proposal.token});
  });
  el('agentAutoReject').addEventListener('click',()=>action('reject'));
  fetch('/api/agent/auto/status').then(r=>r.json()).then(d=>{
    const detail = d.provider==='hf_local_adapter'
      ? ('Lokaler NERAL-LoRA-Adapter · '+(d.confirmed_available?'geladen':'noch nicht geladen'))
      : 'Ollama lokal · kein verifizierter NERAL-Adapter';
    el('agentAutoModel').textContent='Lokales Modell: '+d.model+' · '+detail;
  }).catch(()=>{el('agentAutoModel').textContent='Modellstatus nicht verfügbar.';});
})();
