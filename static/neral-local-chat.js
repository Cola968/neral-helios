'use strict';
// Local same-origin NERAL 7B chat. Never save prompts in browser storage;
// never send them to another provider or execute returned code.
(() => {
  const el = id => document.getElementById(id);
  if (!el('neralLocalChatForm')) return;
  let messages = [];
  let busy = false;
  let compareAvailable = false;
  const encoder = new TextEncoder();

  const identity = el('neralLocalChatIdentity');
  const error = el('neralLocalChatError');
  const transcript = el('neralLocalChatAnswer');
  const form = el('neralLocalChatForm');
  const prompt = el('neralLocalChatPrompt');
  const thinkSwitch = el('neralLocalChatThink');
  const compareButton = el('neralLocalChatCompare');
  const comparePanel = el('neralComparePanel');
  const compareBase = el('neralCompareBase');
  const compareNeral = el('neralCompareNeral');
  const compareStatus = el('neralCompareStatus');

  function updateIdentity(data) {
    compareAvailable = data.compare_available === true;
    compareButton.disabled = busy || !compareAvailable;
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
      role.textContent = item.role === 'user' ? 'Du: ' :
        item.tool_used === 'neral_math_exact' ? 'NERAL (exakte lokale Berechnung): ' :
        item.mode === 'THINK' && item.verified === true ? 'NERAL THINK (Endwert geprüft): ' :
        item.mode === 'THINK' ? 'NERAL THINK (ungeprüft): ' : 'NERAL: ';
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
      body:JSON.stringify(data), signal:AbortSignal.timeout(path === 'think' ? 180000 : 120000)
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
    compareButton.disabled = true;
    error.textContent = '';
    transcript.setAttribute('aria-busy','true');
    try {
      // Bound each history item by UTF-8 bytes and drop oldest COMPLETE
      // turns when the aggregate prompt would exceed the local API budget.
      let history = messages.slice(-12).map(m => {
        let content = m.content.slice(0,1700);
        while (encoder.encode(content).length > 1900) content = content.slice(0,-1);
        return {role:m.role,content};
      });
      while (history.length && encoder.encode(JSON.stringify(history)).length > 9000) {
        history = history.slice(2);
      }
      const answer = await req(thinkSwitch && thinkSwitch.checked ? 'think' : 'generate',{prompt:text,history});
      messages.push({role:'user',content:text},{role:'assistant',content:answer.text,
        mode: answer.mode === 'THINK' ? 'THINK' : null,
        verified: answer.verified === true && answer.verification_scope === 'numeric_final_only',
        tool_used: answer.model_generated === false && answer.tool_used === 'neral_math_exact'
          ? 'neral_math_exact' : null});
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
      compareButton.disabled = !compareAvailable;
    }
  });

  compareButton.addEventListener('click', async () => {
    if (busy || !compareAvailable) return;
    const text = prompt.value.trim();
    if (!text || encoder.encode(text).length > 4096) {
      error.textContent = 'Für den A/B-Vergleich eine Eingabe mit höchstens 4096 UTF-8-Bytes eintragen.';
      return;
    }
    busy = true;
    compareButton.disabled = true;
    el('neralLocalChatSend').disabled = true;
    error.textContent = '';
    comparePanel.hidden = false;
    compareBase.textContent = 'Lokale Foundation berechnet …';
    compareNeral.textContent = 'NERAL-LoRA berechnet …';
    compareStatus.textContent = 'Die beiden Varianten werden nacheinander auf derselben GPU berechnet.';
    comparePanel.setAttribute('aria-busy', 'true');
    try {
      // One-shot A/B: no previous chat turns, tools or different system prompts.
      const result = await req('compare', {prompt:text});
      if (result.model_generated !== true || result.same_prompt !== true ||
          result.same_decoding !== true || result.adapter_loaded !== true ||
          typeof result.base !== 'string' || typeof result.neral !== 'string') {
        throw new Error('Vergleich ohne nachgewiesene Adapter-Herkunft abgelehnt.');
      }
      compareBase.textContent = result.base;
      compareNeral.textContent = result.neral;
      compareStatus.textContent = result.base === result.neral
        ? 'Identische Antworten. Kein Unterschied nachgewiesen; keine Qualitätsbewertung.'
        : 'Unterschiedliche Antworten. Welche besser ist, muss unabhängig geprüft werden.';
      updateIdentity(result);
    } catch (cause) {
      comparePanel.hidden = true;
      compareBase.textContent = '';
      compareNeral.textContent = '';
      compareStatus.textContent = '';
      error.textContent = cause.message || 'A/B-Vergleich fehlgeschlagen.';
    } finally {
      comparePanel.removeAttribute('aria-busy');
      busy = false;
      el('neralLocalChatSend').disabled = false;
      compareButton.disabled = !compareAvailable;
    }
  });

  el('neralLocalChatClear').addEventListener('click', () => {
    if (busy) return;
    messages = [];
    prompt.value = '';
    error.textContent = '';
    comparePanel.hidden = true;
    compareBase.textContent = '';
    compareNeral.textContent = '';
    compareStatus.textContent = '';
    render();
  });

  req('status').then(updateIdentity).catch(() => {
    identity.textContent = 'Lokale Modellinformationen nicht erreichbar.';
  });
})();
