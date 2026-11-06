'use strict';
// Separate local-only agent UI: no eval, innerHTML, shell, remote API or credentials.
(() => {
  const el = id => document.getElementById(id);
  const panel = el('agentPanel');
  const btn = el('agentView');
  if (!panel || !btn) return;
  let current = null, approval = null, working = false, previous = [];
  const surfaces = [el('welcome'), el('log'), document.querySelector('.composer-wrap'),
                    el('projectWorkspace')].filter(Boolean);

  function show() {
    if (panel.hidden) {
      previous = surfaces.map(node => [node, node.hidden]);
      surfaces.forEach(node => { node.hidden = true; });
      panel.hidden = false;
      btn.classList.add('selected');
      el('chatView')?.classList.remove('selected');
      el('projectShortcut')?.classList.remove('selected');
    }
  }
  function close() {
    if (panel.hidden) return;
    panel.hidden = true;
    previous.forEach(([node, hidden]) => { node.hidden = hidden; });
    previous = [];
    btn.classList.remove('selected');
  }
  btn.addEventListener('click', show);
  el('agentClose').addEventListener('click', close);
  for (const id of ['chatView', 'projectShortcut', 'openProjects', 'welcomeProjects', 'closeProjects']) {
    // Close before the existing view handler toggles its own surfaces.
    el(id)?.addEventListener('click', close, true);
  }

  async function request(path, payload) {
    const opts = {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload || {})};
    let response;
    try { response = await fetch(path, opts); } catch { throw Error('Lokaler Server nicht erreichbar.'); }
    let value;
    try { value = await response.json(); } catch { throw Error('Ungültige Serverantwort.'); }
    if (!response.ok) throw Error(value.error || 'Agentenaktion fehlgeschlagen.');
    return value;
  }
  function update(data) {
    current = data;
    const status = data.status;
    const running = status === 'running', paused = status === 'paused';
    el('agentNext').disabled = !running;
    el('agentAll').disabled = !running || working;
    el('agentPause').disabled = !running;
    el('agentResume').disabled = !paused;
    el('agentCancel').disabled = !['running', 'paused', 'awaiting_approval'].includes(status);
    el('agentPreview').disabled = !running;
    el('agentUndo').disabled = !data.can_undo;
    el('agentApprove').disabled = status !== 'awaiting_approval' || !approval;
    el('agentState').textContent = status + ' · ' + data.steps_completed + '/' + data.plan.length +
      ' Schritte · ' + data.goal + (data.error ? ' · ' + data.error : '');
    const visible = {plan:data.plan, results:data.results, events:data.events};
    el('agentEvidence').textContent = JSON.stringify(visible, null, 2);
    if (data.proposal) el('agentDiff').textContent = data.proposal.diff || '(Keine Textunterschiede)';
  }
  function message(error) {
    el('agentState').textContent = error.message || String(error);
  }
  async function action(name, payload) {
    if (!current) return;
    try {
      update(await request('/api/agent/' + current.id + '/' + name, payload));
    } catch (error) { message(error); }
  }
  el('agentForm').addEventListener('submit', async event => {
    event.preventDefault();
    approval = null;
    el('agentDiff').textContent = '';
    try {
      update(await request('/api/agent/start', {
        goal:el('agentGoal').value, strategy:el('agentStrategy').value,
        path:el('agentPath').value || '.', query:el('agentQuery').value
      }));
    } catch (error) { message(error); }
  });
  el('agentNext').addEventListener('click', () => action('next'));
  el('agentPause').addEventListener('click', () => action('pause'));
  el('agentResume').addEventListener('click', () => action('resume'));
  el('agentCancel').addEventListener('click', () => { approval = null; action('cancel'); });
  el('agentUndo').addEventListener('click', () => action('undo'));
  el('agentAll').addEventListener('click', async () => {
    if (!current || working) return;
    working = true;
    el('agentAll').disabled = true;
    try {
      for (let i=0; i<12 && current?.status === 'running'; i++) {
        update(await request('/api/agent/' + current.id + '/next', {}));
      }
    } catch (error) { message(error); }
    finally { working = false; if (current) update(current); }
  });
  el('agentPreview').addEventListener('click', async () => {
    if (!current || current.status !== 'running') return;
    approval = null;
    try {
      const path = el('agentEditPath').value;
      const info = await request('/api/agent/check', {path});
      const data = await request('/api/agent/' + current.id + '/propose', {
        path, content: el('agentEditContent').value, expected_sha:info.sha256
      });
      approval = data.proposal?.token || null;
      update(data);
    } catch (error) { message(error); }
  });
  el('agentApprove').addEventListener('click', async () => {
    if (!approval || !current) return;
    if (!window.confirm('Geprüften Datei-Diff tatsächlich lokal schreiben?')) return;
    const token = approval;
    approval = null; // One use only
    await action('approve', {token});
    el('agentDiff').textContent = '';
  });
  el('agentEditPath').addEventListener('input', () => { approval=null; el('agentApprove').disabled=true; });
  el('agentEditContent').addEventListener('input', () => { approval=null; el('agentApprove').disabled=true; });
})();
