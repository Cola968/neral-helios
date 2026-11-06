'use strict';
/* Explicit local-only Computer Use. No screenshot is taken without a click.
   The model cannot execute these actions and no image is uploaded. */
(() => {
  const get = name => document.getElementById(name);
  if (!get('computerStart')) return;
  let enabled = false;
  let pending = null;
  let screen = null;

  async function post(path, payload = {}) {
    let r;
    try {
      r = await fetch('/api/computer/' + path, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify(payload)
      });
    } catch { throw new Error('Lokaler NERAL-PC nicht erreichbar.'); }
    let data;
    try { data = await r.json(); } catch { throw new Error('Ungültige lokale Serverantwort.'); }
    if (!r.ok) throw new Error(data.error || 'Computeraktion fehlgeschlagen.');
    return data;
  }
  const announce = message => {get('computerStatus').textContent = message;};
  const clearPending = () => {
    pending = null;
    get('computerApprove').disabled = true;
    get('computerProposed').textContent = '';
  };
  function reflect(status) {
    enabled = !!status.enabled;
    get('computerStart').disabled = enabled;
    get('computerScreenshot').disabled = !enabled;
    get('computerStop').disabled = !enabled;
    get('computerPreview').disabled = !enabled;
    if (!enabled) {
      clearPending();
      get('computerFrame').hidden = true;
      screen = null;
    }
    announce(enabled
      ? 'Computer Use lokal aktiv. Kein automatischer Modellzugriff. Freigaben gelten für Einzelaktionen.'
      : 'Computer Use nicht aktiv. Benötigt NERAL_ENABLE_COMPUTER_USE=1 auf deinem PC.');
  }
  async function operation(fn) {
    try { await fn(); } catch (err) { announce(err.message || String(err)); clearPending(); }
  }
  get('computerStart').addEventListener('click', () => operation(async () => {
    if (!window.confirm('Computer Use nur auf diesem PC für fünf Minuten aktivieren? Bildschirm bleibt lokal.')) return;
    reflect(await post('start'));
  }));
  get('computerStop').addEventListener('click', () => operation(async () => {
    reflect(await post('stop'));
  }));
  get('computerScreenshot').addEventListener('click', () => operation(async () => {
    clearPending();
    let result = await post('screenshot');
    screen = {width:result.width,height:result.height};
    get('computerFrame').src = 'data:' + result.media_type + ';base64,' + result.image_base64;
    get('computerFrame').hidden = false;
    announce('Bildschirmaufnahme nur lokal angezeigt. Klicke auf das Bild, um Koordinaten zu wählen.');
  }));
  get('computerFrame').addEventListener('click', e => {
    if (!enabled || !screen || !get('computerFrame').getBoundingClientRect) return;
    const bounds = get('computerFrame').getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    get('computerX').value = String(Math.min(screen.width - 1, Math.max(0,
      Math.floor((e.clientX - bounds.left) / bounds.width * screen.width))));
    get('computerY').value = String(Math.min(screen.height - 1, Math.max(0,
      Math.floor((e.clientY - bounds.top) / bounds.height * screen.height))));
    clearPending();
  });
  function payload() {
    const action = get('computerAction').value;
    switch (action) {
      case 'click':
      case 'move':
        return {action,x:Number(get('computerX').value),y:Number(get('computerY').value)};
      case 'scroll':
        return {action,amount:Number(get('computerScroll').value)};
      case 'type':
        return {action,text:get('computerText').value};
      case 'press':
        return {action,key:get('computerKey').value};
      default: throw new Error('Unbekannte Aktion.');
    }
  }
  get('computerPreview').addEventListener('click', () => operation(async () => {
    if (!enabled) return;
    clearPending();
    const data = await post('preview',payload());
    pending = data.approval_token;
    get('computerProposed').textContent = data.summary
      + ' – einzeln bestätigen, Freigabe läuft nach 20 Sekunden ab.';
    get('computerApprove').disabled = false;
  }));
  get('computerApprove').addEventListener('click', () => operation(async () => {
    if (!pending || !enabled) return;
    if (!window.confirm('Diese einzelne lokale Computeraktion jetzt wirklich ausführen?')) return;
    const token = pending;
    clearPending();
    const data = await post('execute',{approval_token:token});
    announce(data.executed ? 'Aktion ausgeführt. Bildschirm manuell neu laden.' : 'Nicht ausgeführt.');
    // Never automatically take another screenshot after acting.
  }));
  for (const name of ['computerAction','computerX','computerY','computerScroll','computerText','computerKey']) {
    get(name).addEventListener('input',clearPending);
    get(name).addEventListener('change',clearPending);
  }
  post('status').then(reflect).catch(() => announce('Computer Use ist nicht verfügbar.'));
})();