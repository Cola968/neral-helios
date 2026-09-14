'use strict';
(() => {
  const bridge = {
    selected: false,
    available: false,
    model: 'claude-opus-5',
    label: 'Claude Opus 5',
    localReady: modelReady,
    localMaxContext: maxContext,
  };
  try { bridge.selected = localStorage.getItem('neral.helios.chat-provider') === 'experiential'; } catch {}

  const localStats = stats;
  const localSyncModelPicker = syncModelPicker;
  const localSetBusy = setBusy;
  const localSetMode = setMode;
  const localGenerateRequest = generateRequest;
  const localRender = render;

  function persistProvider() {
    try { localStorage.setItem('neral.helios.chat-provider', bridge.selected ? 'experiential' : 'local'); } catch {}
  }

  function updateChrome() {
    if (!$('modelPicker')) return;
    const heading = $('modelPicker').querySelector('.model-menu-heading');
    if (heading) heading.textContent = 'Modelle';
    if (bridge.selected) {
      $('modelName').textContent = 'Claude Opus 5 · Cloud';
      $('modelPickerNote').textContent = bridge.available
        ? 'Claude Opus 5 · über Experiential Labs · API-Key bleibt im lokalen Backend'
        : 'Claude Opus 5 ist noch nicht eingerichtet. EXPLABS_API_KEY fehlt.';
      $('modeHint').textContent = 'Claude Opus 5 · Cloud über Experiential Labs';
      $('footnote').textContent = 'Cloud-Modell über Experiential Labs. Antworten können Fehler machen.';
    } else {
      $('footnote').textContent = 'Neral kann Fehler machen.';
    }
  }

  function selectCloud(value, redraw = true) {
    bridge.selected = Boolean(value && bridge.available);
    modelReady = bridge.selected ? true : bridge.localReady;
    maxContext = bridge.selected ? 1_000_000 : bridge.localMaxContext;
    persistProvider();
    if (redraw) {
      syncModelPicker();
      setMode(mode);
      setBusy(busy);
      render();
    }
  }

  syncModelPicker = function () {
    // Render the local choices with their true local readiness, then restore the
    // cloud selection state. This prevents a cloud selection from making a
    // local checkpoint look loaded when it is not.
    const effectiveReady = modelReady;
    modelReady = bridge.localReady;
    localSyncModelPicker();
    modelReady = bridge.selected ? true : bridge.localReady;

    for (const button of $('modelOptions').querySelectorAll('button[data-checkpoint]')) {
      const originalClick = button.onclick;
      button.onclick = event => {
        bridge.selected = false;
        modelReady = bridge.localReady;
        maxContext = bridge.localMaxContext;
        persistProvider();
        updateChrome();
        return originalClick?.call(button, event);
      };
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.cloudModel = bridge.model;
    button.setAttribute('aria-pressed', String(bridge.selected));
    button.disabled = busy || !ready || !bridge.available;

    const title = document.createElement('span');
    title.className = 'model-option-title';
    title.textContent = bridge.label;
    const version = document.createElement('span');
    version.className = 'model-version';
    version.textContent = 'Experiential Labs' + (bridge.selected ? ' · aktiv' : '');
    const description = document.createElement('small');
    description.textContent = bridge.available
      ? 'Cloud · stark für UI, Code und komplexe Aufgaben'
      : 'Nicht bereit · EXPLABS_API_KEY setzen und Helios neu starten';
    button.append(title, version, description);
    button.onclick = () => {
      if (busy || !ready || !bridge.available) return;
      $('modelPicker').open = false;
      selectCloud(true);
      notify('Claude Opus 5 ausgewählt. Anfragen laufen jetzt über Experiential Labs.');
    };
    $('modelOptions').append(button);

    if (bridge.selected) {
      modelReady = true;
      maxContext = 1_000_000;
    } else {
      modelReady = bridge.localReady;
      maxContext = bridge.localMaxContext;
    }
    updateChrome();
  };

  stats = function (info) {
    bridge.localReady = info.checkpoint_loaded === true;
    bridge.localMaxContext = Number(info.max_seq_len) || 4096;
    const cloud = Array.isArray(info.cloud_models)
      ? info.cloud_models.find(item => item && item.id === 'claude-opus-5')
      : null;
    bridge.available = Boolean(cloud?.available);
    if (cloud?.id) bridge.model = cloud.id;
    if (cloud?.label) bridge.label = cloud.label;
    if (!bridge.available) bridge.selected = false;

    localStats(info);
    modelReady = bridge.selected ? true : bridge.localReady;
    maxContext = bridge.selected ? 1_000_000 : bridge.localMaxContext;
    persistProvider();
    syncModelPicker();
    setBusy(busy);
  };

  setBusy = function (value) {
    localSetBusy(value);
    $('send').disabled = value || !ready || (mode === 'chat' && !(bridge.selected ? bridge.available : bridge.localReady));
    const cloudButton = $('modelOptions').querySelector('button[data-cloud-model]');
    if (cloudButton) cloudButton.disabled = value || !ready || !bridge.available;
    updateChrome();
  };

  setMode = function (value) {
    localSetMode(value);
    if (value === 'chat' && bridge.selected) {
      $('modeHint').textContent = 'Claude Opus 5 · Cloud über Experiential Labs';
    }
    updateChrome();
  };

  function cloudHistory(messages, prompt) {
    const encoder = new TextEncoder();
    const limit = 4096 - encoder.encode(prompt).length;
    if (limit <= 0) return [];

    const pairs = [];
    for (let i = 0; i + 1 < messages.length; i += 2) {
      const user = messages[i], assistant = messages[i + 1];
      if (user?.role !== 'user' || assistant?.role !== 'assistant') continue;
      pairs.push([
        { role: 'user', content: String(user.content || '') },
        { role: 'assistant', content: String(assistant.content || '') },
      ]);
    }

    const selected = [];
    let bytes = 0;
    for (let i = pairs.length - 1; i >= 0 && selected.length < 6; i--) {
      const pairBytes = encoder.encode(pairs[i][0].content).length + encoder.encode(pairs[i][1].content).length;
      if (bytes + pairBytes > limit) continue;
      selected.unshift(pairs[i]);
      bytes += pairBytes;
    }
    return selected.flat();
  }

  generateRequest = async function (payload, onProgress) {
    if (payload.mode !== 'chat' || !bridge.selected) {
      return localGenerateRequest(payload, onProgress);
    }
    const pendingText = document.querySelector('#log .message.pending .content');
    if (pendingText) pendingText.textContent = 'Claude Opus 5 denkt …';

    // The submit handler has already appended the current user message. Send
    // only earlier complete pairs as history; the current prompt is separate.
    const previous = Array.isArray(current?.messages) ? current.messages.slice(0, -1) : [];
    return localGenerateRequest({
      ...payload,
      provider: 'experiential',
      model: bridge.model,
      history: cloudHistory(previous, payload.prompt),
    }, onProgress);
  };

  render = function () {
    localRender();
    const nodes = [...$('log').querySelectorAll('article.message')];
    for (let i = 0; i < (current?.messages?.length || 0); i++) {
      const message = current.messages[i];
      if (message?.role !== 'assistant' || message.mode !== 'cloud-generation') continue;
      const label = nodes[i]?.querySelector('.message-label');
      if (label) label.textContent = '◈ Claude · Opus 5 · Experiential';
      const meta = nodes[i]?.querySelector('.meta');
      if (meta && meta.textContent.startsWith('0 Byte-Token')) meta.textContent = 'Claude Opus 5 · Cloud-Antwort';
    }
    updateChrome();
  };

  // Refresh once after this bridge is installed so a very fast initial /api/info
  // response cannot leave the cloud model undiscovered.
  jsonRequest('/api/info').then(info => stats(info)).catch(() => {});
})();
