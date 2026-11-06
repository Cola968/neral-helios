'use strict';
// Local same-origin NERAL 7B chat. Never save prompts in browser storage;
// never send them to another provider or execute returned code.
(() => {
  const el = id => document.getElementById(id);
  if (!el('neralLocalChatForm')) return;
  let messages = [];
  let busy = false;
  const encoder = new TextEncoder();

  const identity = el('neralLocalChatIdentity');
  const error = el('neralLocalChatError');
  const transcript = el('neralLocalChatAnswer');
  const form = el('neralLocalChatForm');
  const prompt = el('neralLocalChatPrompt');

  function updateIdentity(data) {
    if (data.adapter_loaded === true) {
      identity.textContent = 'NERAL-LoRA lokal geladen (experimentell) · Qwen2.5-Coder-7B-Foundation · Adapter SHA-256: ' +
        (data.adapter_sha256 || 'unbekannt') + ' · kein nachgewiesener Qualitätsvorsprung.';
    } else if (data.model_origin === 'local_adapter_configured_not_loaded') {
      identity.textContent = 'NERAL-LoRA lokal konfiguriert, noch nicht geladen. ' +
        'Eine erfolgreiche Inferenz bestätigt erst die tatsächliche Aktivierung.';
    } else {
      identity.textContent = 'Nur lokales Basismodell: ' + String(data.model || 'unbekannt') +
        '. Kein verifizierter NERAL-LoRA-Adapter aktiv.';
    }
  }

  function render() {
    transcript.replaceChildren();
    if (!messages.length) {
      transcript.textContent = 'Noch keine Nachricht. Der Verlauf wird nur im Arbeitsspeicher dieses Browserfensters gehalten.';
      return;
    }
    for (const item of messages) {
      const line = document.createElement('p');
      const role = document.createElement('strong');
      role.textContent = item.role === 'user' ? 'Du: ' : 'NERAL: ';
      const content = document.createElement('span');
      content.textContent = item.content;
      line.append(role, content);
      transcript.append(line);
    }
    transcript.scrollTop = transcript.scrollHeight;
  }

  async function req(path, data) {
    const init = data === undefined ? {method:'GET',signal:AbortSignal.timeout(8000)} : {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(data), signal:AbortSignal.timeout(120000)
    };
    const response = await fetch('/api/neral/chat/' + path, init);
    let obj;
    try {obj = await response.json();}
    catch {throw new Error('Ungültige lokale Modellantwort.');}
    if (!response.ok) throw new Error(obj.error || 'NERAL ist lokal nicht verfügbar.');
    return obj;
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (busy) return;
    const text = prompt.value.trim();
    if (!text || encoder.encode(text).length > 4096) {
      error.textContent = 'Eingabe muss zwischen 1 und 4096 UTF-8-Bytes enthalten.';
      return;
    }
    busy = true;
    el('neralLocalChatSend').disabled = true;
    error.textContent = '';
    transcript.setAttribute('aria-busy','true');
    try {
      const history = messages.slice(-12).map(m => ({
        role:m.role,
        content:m.content.slice(0,1900),
      }));
      const answer = await req('generate',{prompt:text,history});
      messages.push({role:'user',content:text},{role:'assistant',content:answer.text});
      // Keep memory bounded and keep alternating user-assistant turns.
      if (messages.length > 12) messages = messages.slice(-12);
      updateIdentity(answer);
      prompt.value = '';
      render();
    } catch (cause) {
      error.textContent = cause.message || 'Lokale Inferenz fehlgeschlagen.';
    } finally {
      transcript.removeAttribute('aria-busy');
      busy = false;
      el('neralLocalChatSend').disabled = false;
    }
  });

  el('neralLocalChatClear').addEventListener('click', () => {
    if (busy) return;
    messages = [];
    prompt.value = '';
    error.textContent = '';
    render();
  });

  req('status').then(updateIdentity).catch(() => {
    identity.textContent = 'Lokale Modellinformationen nicht erreichbar.';
  });
})();
