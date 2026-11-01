'use strict';
const CACHE='neral-shell-v3';
const SHELL=['/','/static/neral.css','/static/chat-experience.css','/static/neral.js','/static/helios-core.js','/static/projects.js','/static/install.js','/static/workspace.js','/static/icon.svg'];
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL))));
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k.startsWith('neral-shell-')&&k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',e=>{const u=new URL(e.request.url);if(e.request.method!=='GET'||u.origin!==self.location.origin||!SHELL.includes(u.pathname))return;const cacheKey=u.pathname;e.respondWith(fetch(e.request).then(r=>{if(r.ok){const copy=r.clone();e.waitUntil(caches.open(CACHE).then(c=>c.put(cacheKey,copy)));}return r;}).catch(()=>caches.match(cacheKey)));});
