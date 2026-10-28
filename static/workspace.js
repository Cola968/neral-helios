'use strict';
let settings={stream:true,sync:true,font:'normal'},syncTimer=null,syncRunning=false,crashCount=0;
try{const saved=JSON.parse(localStorage.getItem('neral.settings')||'{}');for(const k of ['stream','sync'])if(typeof saved[k]==='boolean')settings[k]=saved[k];if(saved.font==='large')settings.font='large';}catch{}
document.documentElement.dataset.font=settings.font;
function persistSettings(){try{localStorage.setItem('neral.settings',JSON.stringify(settings));}catch{notify('Einstellungen konnten nicht gespeichert werden.');}}
async function generateRequest(payload,onProgress){
 try{
  const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...payload,stream:settings.stream&&payload.mode==='chat'}),signal:AbortSignal.timeout(150000)});
  if(!response.ok){let data={};try{data=await response.json();}catch{}const error=Error(data.error||'Anfrage fehlgeschlagen. Bitte erneut versuchen.');error.status=response.status;throw error;}
  return await HeliosCore.readGenerationResponse(response,onProgress);
 }catch(error){
  if(error.name==='AbortError'||error.name==='TimeoutError')throw Error('Die Antwort hat zu lange gedauert. Dein Entwurf bleibt erhalten.');
  if(error instanceof TypeError)throw Error('Verbindung zum lokalen Server verloren. Prüfe die Verbindung und versuche es erneut.');
  throw error;
 }
}
function deletedQueue(){try{return JSON.parse(localStorage.getItem('neral.sync.deleted')||'{}');}catch{return {};}}
function queueChatDelete(chat){if(chat.temporary)return;const pending=deletedQueue();pending[chat.id]=chat._revision||0;try{localStorage.setItem('neral.sync.deleted',JSON.stringify(pending));}catch{}scheduleSync();}
function cleanChat(c){return {id:c.id,title:c.title,messages:c.messages,draft:c.draft||'',mode:c.mode||'chat',imported:!!c.imported,temporary:false};}
function saveSynced(){try{localStorage.setItem(storageKey,JSON.stringify(chats));}catch{notify('Lokaler Speicher voll. Chats bitte exportieren.');}}
function scheduleSync(){clearTimeout(syncTimer);syncTimer=setTimeout(()=>syncChats(),1600);}
async function syncChats(manual=false){
 if(!settings.sync||syncRunning||busy||!navigator.onLine)return;
 syncRunning=true;$('syncStatus').textContent=' · Speichern …';
 try{
  const deletions=deletedQueue(),processedDeletes={...deletions};
  for(const [id,revision] of Object.entries(deletions).slice(0,20)){
   try{await jsonRequest('/api/chats/'+id,{revision,deleted:true});delete deletions[id];}
   catch(e){if(e.status===409){delete deletions[id];notify('Ein inzwischen geänderter Chat wurde nicht gelöscht. Bitte prüfen.');}else throw e;}
  }
  const latestDeletes=deletedQueue();for(const [id,revision] of Object.entries(latestDeletes)){if(!Object.hasOwn(deletions,id)&&processedDeletes[id]===revision)delete latestDeletes[id];}localStorage.setItem('neral.sync.deleted',JSON.stringify(latestDeletes));
  for(const c of chats.filter(c=>!c.temporary&&!c._syncBlocked&&(c._dirty||c._revision===undefined)).slice(0,20)){
   const snapshot=JSON.stringify(cleanChat(c));
   try{const r=await jsonRequest('/api/chats/'+c.id,{revision:c._revision||0,chat:JSON.parse(snapshot)});c._revision=r.revision;if(!chats.includes(c)){const queued=deletedQueue();if(Object.hasOwn(queued,c.id)){queued[c.id]=r.revision;localStorage.setItem('neral.sync.deleted',JSON.stringify(queued));}}if(snapshot===JSON.stringify(cleanChat(c)))c._dirty=false;}
   catch(e){if(e.status===409){c.id=crypto.randomUUID();c.title=(c.title+' · Konfliktkopie').slice(0,100);c._revision=0;c._dirty=true;notify('Gleichzeitige Änderung: Deine Fassung bleibt als Konfliktkopie erhalten.');}else if(e.status===400||e.status===413){c._syncBlocked=true;notify('Ein Chat bleibt lokal: '+e.message);}else throw e;}
  }
  const data=await jsonRequest('/api/chats');
  for(const remote of data.chats){if(Object.hasOwn(deletedQueue(),remote.id))continue;const local=chats.find(c=>c.id===remote.id);if(local?._dirty)continue;if(remote.deleted){chats=chats.filter(c=>c.id!==remote.id);if(current?.id===remote.id){current=null;fresh();}continue;}
   if(!local){chats.push({...remote.chat,_revision:remote.revision,_dirty:false});continue;}
   if((local._revision||0)<remote.revision){Object.assign(local,remote.chat,{_revision:remote.revision,_dirty:false});if(current?.id===local.id){$('prompt').value=local.draft||'';setMode(local.mode||'chat');}}
  }
  saveSynced();render();$('syncStatus').textContent=' · Gespeichert';if(manual)$('settingsStatus').textContent='Chats mit dem PC abgeglichen.';
 }catch(e){$('syncStatus').textContent=e.status===401?' · Neu verbinden':' · Lokal gespeichert';if(manual)$('settingsStatus').textContent=e.message;}
 finally{syncRunning=false;saveSynced();}
}
window.addEventListener('neral:changed',scheduleSync);
window.addEventListener('online',scheduleSync);
setInterval(()=>{if(document.visibilityState==='visible')syncChats();},30000);
scheduleSync();
$('openSettings').onclick=()=>{$('settingTheme').value=document.documentElement.dataset.theme||'dark';$('settingFont').value=settings.font;$('settingStream').checked=settings.stream;$('settingSync').checked=settings.sync;$('settingsDialog').showModal();};
$('closeSettings').onclick=()=>$('settingsDialog').close();
$('settingTheme').onchange=()=>{document.documentElement.dataset.theme=$('settingTheme').value;try{localStorage.setItem('neral.theme',$('settingTheme').value);}catch{}};
$('settingFont').onchange=()=>{settings.font=$('settingFont').value;document.documentElement.dataset.font=settings.font;persistSettings();};
$('settingStream').onchange=()=>{settings.stream=$('settingStream').checked;persistSettings();};
$('settingSync').onchange=()=>{settings.sync=$('settingSync').checked;persistSettings();if(settings.sync)syncChats(true);};
$('syncNow').onclick=()=>syncChats(true);$('reloadApp').onclick=()=>location.reload();
function quickAccess(){if(busy)return;$('settingsDialog').close();projectView(false);closeSidebar();$('prompt').focus();}
$('quickAccess').onclick=quickAccess;
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.shiftKey&&e.code==='Space'){e.preventDefault();quickAccess();}});
const launch=new URLSearchParams(location.search);if(launch.get('quick')==='solve'){setMode('solve');$('prompt').focus();}else if(launch.has('quick'))quickAccess();
function reportCrash(){crashCount++;notify('Ein App-Fehler ist aufgetreten. Entwurf bleibt lokal. Über Einstellungen → App neu verbinden fortsetzen.');}
window.addEventListener('error',reportCrash);window.addEventListener('unhandledrejection',reportCrash);
$('diagnostics').onclick=()=>{const data={app:'Neral workspace v2',online:navigator.onLine,model:activeModel,modelReady,preparationMode,streaming:settings.stream,sync:settings.sync,crashCount};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='neral-diagnose.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
