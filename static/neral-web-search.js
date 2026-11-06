'use strict';
// Explicit query only; no prompt/history upload or untrusted HTML insertion.
(() => {
  const el = id => document.getElementById(id);
  if (!el('webSearchToggle') || !el('webSearchPanel')) return;
  const panel = el('webSearchPanel'), toggle = el('webSearchToggle');
  const form = el('webSearchForm'), query = el('webSearchQuery');
  const results = el('webSearchResults'), status = el('webSearchStatus');
  let busy = false;

  toggle.addEventListener('click', () => {
    panel.hidden = !panel.hidden;
    toggle.setAttribute('aria-expanded', String(!panel.hidden));
    if (!panel.hidden) query.focus();
  });
  el('webSearchClose').addEventListener('click', () => {
    panel.hidden = true;
    toggle.setAttribute('aria-expanded', 'false');
    toggle.focus();
  });

  function isPublicHttps(address) {
    try {
      const url = new URL(address);
      if (url.protocol !== 'https:' || url.username || url.password || url.port && url.port !== '443') return false;
      if (/^(localhost|.*\.local|.*\.localhost|.*\.internal)$/i.test(url.hostname)) return false;
      if (!url.hostname.includes('.') || /^(\d{1,3}\.){3}\d{1,3}$/.test(url.hostname)) return false;
      return true;
    } catch { return false; }
  }

  function showRows(data) {
    results.replaceChildren();
    const rows = Array.isArray(data.results) ? data.results.slice(0, 5) : [];
    for (const item of rows) {
      if (!item || !isPublicHttps(item.url) || typeof item.title !== 'string') continue;
      const article = document.createElement('article');
      article.className = 'web-result';
      const link = document.createElement('a');
      link.href = item.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
      link.textContent = item.title.slice(0, 180);
      const domain = document.createElement('small');
      domain.textContent = new URL(item.url).hostname;
      const summary = document.createElement('p');
      summary.textContent = String(item.snippet || '').slice(0, 350);
      article.append(link, domain, summary);
      results.append(article);
    }
    status.textContent = results.children.length
      ? results.children.length + ' externe Suchtreffer von ' + String(data.provider || 'Suchanbieter') + ' · Inhalte nicht verifiziert.'
      : 'Keine verarbeitbaren Treffer. Der Anbieter könnte die Suche eingeschränkt haben.';
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    if (busy) return;
    const text = query.value.trim();
    if (text.length < 2 || text.length > 160) {
      status.textContent = 'Suchbegriff muss 2–160 Zeichen enthalten.';
      return;
    }
    busy = true;
    el('webSearchSubmit').disabled = true;
    status.textContent = 'Suche läuft …';
    results.replaceChildren();
    try {
      const response = await fetch('/api/web/search', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({query:text}), signal:AbortSignal.timeout(12000)
      });
      const data = await response.json();
      if (!response.ok) throw Error(data.error || 'Websuche fehlgeschlagen.');
      showRows(data);
    } catch(error) {
      status.textContent = error.message || 'Websuche nicht erreichbar.';
    } finally {
      busy = false;
      el('webSearchSubmit').disabled = false;
    }
  });
})();
