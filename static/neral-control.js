'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let capabilities = null, requestId = null, busy = false, resultReceived = false;
  let history = [], historyKey = null, submittedPrompt = '';
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
    const key=JSON.stringify([p?.id,info.runtime_id,info.lora_adapter_id,info.lora_weights_version]);
    if (historyKey!==key) { history=[]; historyKey=key; }
    $('runtimeIdentity').replaceChildren();
    for (const [label,key] of [['Quelle','source'],['Anbieter','provider_name'],['Basismodell','base_model_id'],
      ['Basis-Revision','base_model_revision'],['Laufzeit','runtime_id'],
      ['Anbieterschnittstelle','provider_wrapper_version'],['LoRA-Version','lora_weights_version'],
      ['Adapter-ID / Hash','lora_adapter_id'],['Adapter-Integrität geprüft','adapter_integrity_verified']]) {
      const row=document.createElement('div'), term=document.createElement('dt'), value=document.createElement('dd');
      term.textContent=label; value.textContent=typeof info[key]==='boolean' ? (info[key]?'Ja':'Nein') : info[key] ?? 'Keine Angabe'; row.append(term,value); $('runtimeIdentity').append(row);
    }
    $('nativeThink').disabled = busy || !p?.native_think;
    if (!p?.native_think) $('nativeThink').checked=false;
    $('providerNote').textContent = p?.id==='ollama'
      ? p.cancellation==='bounded-cancellable-http'
        ? 'Stopp schließt die lokale HTTP-Verbindung. Ein tatsächlicher Stopp der Serverberechnung bleibt unbestätigt.'
        : 'Dieser Stand prüft Stopp nur kooperativ vor und nach HTTP. Serverstopp ist unbestätigt.'
      : p ? 'Nutzt nur das bereits geladene lokale Modell. Stopp wird zwischen generierten Tokens geprüft.'
        : 'Werkzeuge funktionieren ohne Modellinferenz. Dieses Panel lädt keine Modellgewichte.';
  }
  function setBusy(value) {
    busy=value;
    for (const el of $('controlForm').elements) el.disabled=value;
    $('taskStop').disabled=!value;
    $('refreshCapabilities').disabled=value;
    $('clearControlHistory').disabled=value;
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
      text.textContent=({calculator:'Rechner',code_check:'Statische Codeprüfung',project_files:'Projektdateien (lesend)',web_search:capabilities.search_label || 'Konfigurierte öffentliche Suche'})[tool.name]+(tool.available ? '' : ' · nicht verfügbar'); label.append(input,text); $('toolOptions').append(label);
    }
    $('capabilityStatus').textContent=capabilities.busy ? 'Eine andere Aufgabe oder ein Modellwechsel läuft.' : 'Funktionen aktualisiert. Bereit.';
    identity();
  }
  function event(item) {
    if (item.type==='progress' && !resultReceived) $('taskStage').textContent=item.text || `In Bearbeitung · Schritt ${item.generated_tokens || 0}`;
    if (item.type==='error') throw new Error(item.error || 'Aufgabe unterbrochen.');
    if (item.type==='done') {
      const result=item.result; resultReceived=true;
      $('taskStage').textContent=({success:'Erfolgreich',error:'Fehlgeschlagen',unavailable:'Nicht verfügbar',incomplete:'Unvollständig',cancelled:'Abgebrochen',timeout:'Zeitlimit erreicht',denied:'Nicht erlaubt'})[result.status] || 'Beendet';
      $('taskResult').textContent=result.text || '';
      const yesNo=value=>value===true?'ja':value===false?'nein':'unbestätigt';
      $('taskCancelState').textContent=`Stopp angefordert: ${yesNo(result.cancel_requested)} · bestätigt: ${yesNo(result.cancel_acknowledged)} · Verbindung geschlossen: ${yesNo(result.transport_closed)} · Serverberechnung gestoppt: ${yesNo(result.provider_execution_stopped)} · Abbruchlatenz: ${result.cancel_ack_latency_ms ?? 'unbekannt'} ms`;
      const metrics=result.metrics || {};
      $('taskMeta').textContent=`Quelle: ${result.source || result.provider_identity?.source || 'nicht verfügbar'} · Modus: ${result.profile || $('taskProfile').value} · Werkzeuge: ${(result.tools_used || []).join(', ') || 'keine'} · Freigaben: ${(result.effective_permissions || []).join(', ') || 'keine'} · Modellaufrufe: ${metrics.model_calls ?? 'unbekannt'} · Eingabe-/Ausgabetoken: ${metrics.input_tokens ?? 'unbekannt'} / ${metrics.output_tokens ?? 'unbekannt'}`;
      if (result.completion_status==='complete' && typeof result.history_receipt==='string') {
        history.push({role:'user',content:submittedPrompt},{role:'assistant',content:result.text,
          completion_status:'complete',history_receipt:result.history_receipt});
        history=history.slice(-12);
        $('taskHistoryStatus').textContent='Vollständig abgeschlossen und Herkunft bestätigt · im aktiven Gesprächskontext.';
      } else {
        $('taskHistoryStatus').textContent='Abgebrochen / Unvollständig – nicht im aktiven Modellkontext.';
      }
    }
  }
  $('controlForm').addEventListener('submit', async e => {
    e.preventDefault(); if (busy || !capabilities) return;
    requestId=crypto.randomUUID(); resultReceived=false;
    submittedPrompt=$('taskPrompt').value;
    const chosen=[...document.querySelectorAll('#toolOptions input:checked')];
    const selected=provider();
    const payload={prompt:$('taskPrompt').value, profile:$('taskProfile').value,
      allowed_tools:chosen.map(t=>t.value), permissions:[...new Set(chosen.map(t=>t.dataset.permission))],
      native_think:$('nativeThink').checked, request_id:requestId, stream:true, history};
    if (selected) { payload.provider=selected.id; payload.runtime_id=selected.identity.runtime_id; }
    setBusy(true); $('taskStage').classList.remove('control-error'); $('taskStage').textContent='Wird gestartet…';
    $('taskResult').textContent=''; $('taskMeta').textContent='';
    $('taskCancelState').textContent=''; $('taskHistoryStatus').textContent='';
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
      $('taskStage').textContent=value.cancel_requested ? 'Stopp angefordert. Bestätigung und Verbindungsende stehen noch aus…' : 'Kein aktiver Auftrag mit dieser ID.'; }
    catch(error) { if (!resultReceived && busy && requestId===stoppedId) { showError(error); $('taskStop').disabled=false; } }
  });
  $('taskProvider').addEventListener('change',identity);
  $('clearControlHistory').addEventListener('click',()=>{
    history=[]; $('taskHistoryStatus').textContent='Aktiver Gesprächskontext geleert.';
  });
  $('refreshCapabilities').addEventListener('click',()=>refresh().catch(showError));
  refresh().catch(showError);
})();
