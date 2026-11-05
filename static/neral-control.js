'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let capabilities = null, requestId = null, busy = false, resultReceived = false;
  const provider = () => capabilities?.providers.find(p => p.id === $('taskProvider').value);
  const showError = error => {
    $('taskStage').textContent = error.message || String(error);
    $('taskStage').classList.add('control-error');
  };
  async function json(url, payload) {
    const response = await fetch(url, payload === undefined ? {} : {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `Anfrage fehlgeschlagen (${response.status}).`);
    return value;
  }
  function identity() {
    const p = provider(), info = p?.identity || capabilities?.identity || {};
    $('runtimeIdentity').replaceChildren();
    for (const [label,key] of [['Quelle','source'],['Basismodell','base_model'],['Laufzeit','runtime_id'],
      ['Anbieterschnittstelle','provider_wrapper_version'],['LoRA-Gewichte','lora_weights_version'],['Aktiver Adapter','active_adapter_id']]) {
      const row=document.createElement('div'), term=document.createElement('dt'), value=document.createElement('dd');
      term.textContent=label; value.textContent=info[key] ?? 'Keine Angabe'; row.append(term,value); $('runtimeIdentity').append(row);
    }
    $('nativeThink').disabled = busy || !p?.native_think;
    if (!p?.native_think) $('nativeThink').checked=false;
    $('providerNote').textContent = p?.id==='ollama'
      ? 'Stopp wird vor und nach der Ollama-Anfrage geprüft. Eine laufende HTTP-Anfrage kann nicht unterbrochen werden.'
      : p ? 'Nutzt nur das bereits geladene lokale Modell. Stopp wird zwischen generierten Tokens geprüft.'
        : 'Werkzeuge funktionieren ohne Modellinferenz. Dieses Panel lädt keine Modellgewichte.';
  }
  function setBusy(value) {
    busy=value;
    for (const el of $('controlForm').elements) el.disabled=value;
    $('taskStop').disabled=!value;
    $('refreshCapabilities').disabled=value;
    if (!value) {
      for (const input of document.querySelectorAll('#toolOptions input')) input.disabled=input.dataset.available!=='true';
      identity();
    }
  }
  async function refresh() {
    capabilities=await json('/api/neral/capabilities');
    const previous=$('taskProvider').value;
    $('taskProvider').replaceChildren(new Option('Nur Werkzeuge',''));
    for (const p of capabilities.providers) $('taskProvider').append(new Option(p.identity.base_model || p.id,p.id));
    if (capabilities.providers.some(p=>p.id===previous)) $('taskProvider').value=previous;
    const legend=document.createElement('legend'); legend.textContent='Freigegebene Werkzeuge';
    $('toolOptions').replaceChildren(legend);
    for (const tool of capabilities.tools) {
      const label=document.createElement('label'), input=document.createElement('input'), text=document.createElement('span');
      label.className='control-toggle'; input.type='checkbox'; input.value=tool.name;
      input.dataset.permission=tool.permission; input.dataset.available=String(tool.available);
      input.disabled=!tool.available; input.checked=tool.available && ['calculator','code_check'].includes(tool.name);
      text.textContent=({calculator:'Rechner',code_check:'Statische Codeprüfung',project_files:'Projektdateien (lesend)',web_search:'Websuche (Netzwerk)'})[tool.name]+(tool.available ? '' : ' · nicht verfügbar'); label.append(input,text); $('toolOptions').append(label);
    }
    $('capabilityStatus').textContent=capabilities.busy ? 'Eine andere Aufgabe oder ein Modellwechsel läuft.' : 'Funktionen aktualisiert. Bereit.';
    identity();
  }
  function event(item) {
    if (item.type==='progress') $('taskStage').textContent=item.text || `In Bearbeitung · Schritt ${item.generated_tokens || 0}`;
    if (item.type==='error') throw new Error(item.error || 'Aufgabe unterbrochen.');
    if (item.type==='done') {
      const result=item.result; resultReceived=true;
      $('taskStage').textContent=({success:'Erfolgreich',error:'Fehlgeschlagen',unavailable:'Nicht verfügbar',incomplete:'Unvollständig',cancelled:'Abgebrochen',timeout:'Zeitlimit erreicht',denied:'Nicht erlaubt'})[result.status] || 'Beendet';
      $('taskResult').textContent=result.text || '';
      $('taskMeta').textContent=`Quelle: ${result.source || result.provider_identity?.source || 'nicht verfügbar'} · Werkzeuge: ${(result.tools_used || []).join(', ') || 'keine'}`;
    }
  }
  $('controlForm').addEventListener('submit', async e => {
    e.preventDefault(); if (busy || !capabilities) return;
    requestId=crypto.randomUUID(); resultReceived=false;
    const chosen=[...document.querySelectorAll('#toolOptions input:checked')];
    const selected=provider();
    const payload={prompt:$('taskPrompt').value, profile:$('taskProfile').value,
      allowed_tools:chosen.map(t=>t.value), permissions:[...new Set(chosen.map(t=>t.dataset.permission))],
      native_think:$('nativeThink').checked, request_id:requestId, stream:true};
    if (selected) { payload.provider=selected.id; payload.runtime_id=selected.identity.runtime_id; }
    setBusy(true); $('taskStage').classList.remove('control-error'); $('taskStage').textContent='Wird gestartet…';
    $('taskResult').textContent=''; $('taskMeta').textContent='';
    try {
      const response=await fetch('/api/neral/task',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      if (!response.ok) { const error=await response.json(); throw new Error(error.error || 'Aufgabe fehlgeschlagen.'); }
      const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer='';
      while (true) {
        const {value,done}=await reader.read(); buffer+=decoder.decode(value,{stream:!done});
        let boundary; while ((boundary=buffer.indexOf('\n'))>=0) { const line=buffer.slice(0,boundary); buffer=buffer.slice(boundary+1); if (line.trim()) event(JSON.parse(line)); }
        if (done) { if (buffer.trim()) event(JSON.parse(buffer)); break; }
      }
      if (!resultReceived) throw new Error('Der Antwortstream wurde vorzeitig beendet. Bitte erneut versuchen.');
    } catch (error) { showError(error); }
    finally { requestId=null; setBusy(false); }
  });
  $('taskStop').addEventListener('click',async()=>{
    if (!requestId) return;
    $('taskStop').disabled=true;
    const stoppedId=requestId;
    try { const value=await json('/api/neral/cancel',{request_id:stoppedId});
      if (resultReceived || !busy || requestId!==stoppedId) return;
      $('taskStage').textContent=value.cancelled ? 'Stopp angefordert. Die laufende Operation wird noch beendet…' : 'Die Aufgabe ist bereits beendet.'; }
    catch(error) { if (!resultReceived && busy && requestId===stoppedId) { showError(error); $('taskStop').disabled=false; } }
  });
  $('taskProvider').addEventListener('change',identity);
  $('refreshCapabilities').addEventListener('click',()=>refresh().catch(showError));
  refresh().catch(showError);
})();
