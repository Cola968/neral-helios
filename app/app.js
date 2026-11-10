const state={view:'chat',messages:[],mode:'chat'};
const titles={home:['Home','Dein lokaler KI-Arbeitsbereich'],chat:['Chat','Direkt mit Helios arbeiten'],solve:['Solve','Mathe & Physik strukturiert lösen'],projects:['Projects','Kontext, Dateien und Entscheidungen']};
const views={home:document.getElementById('homeView'),chat:document.getElementById('chatView'),solve:document.getElementById('solveView'),projects:document.getElementById('projectsView')};
const title=document.getElementById('viewTitle'),subtitle=document.getElementById('viewSubtitle'),chatThread=document.getElementById('chatThread'),chatEmpty=document.getElementById('chatEmpty'),input=document.getElementById('composerInput'),toast=document.getElementById('toast'),runtimePanel=document.getElementById('runtimePanel'),scrim=document.getElementById('scrim'),composerState=document.getElementById('composerState');

function showToast(text){toast.textContent=text;toast.classList.add('show');clearTimeout(showToast.t);showToast.t=setTimeout(()=>toast.classList.remove('show'),2200)}
function setView(name){if(!views[name])return;state.view=name;Object.entries(views).forEach(([k,v])=>v.classList.toggle('active',k===name));document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===name));title.textContent=titles[name][0];subtitle.textContent=titles[name][1];if(name==='chat')setTimeout(()=>input.focus(),40)}
function setMode(mode){state.mode=mode;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));composerState.textContent=mode==='chat'?'Neral 50M · Preview':mode==='solve'?'SOLVE · strukturierter Modus':'VERIFY · Prüfmodus'}
function addMessage(role,text){chatEmpty.style.display='none';const el=document.createElement('div');el.className=`message ${role}`;el.textContent=text;chatThread.appendChild(el);state.messages.push({role,text,mode:state.mode});chatThread.scrollTop=chatThread.scrollHeight}
function fakeHeliosReply(prompt){const modeLabel=state.mode==='solve'?'SOLVE':state.mode==='verify'?'VERIFY':'Chat';setTimeout(()=>addMessage('assistant',`${modeLabel}-Preview: Die echte Neral-Runtime ist noch nicht mit diesem neuen App-Shell verbunden. Dein Prompt ist angekommen. Der nächste technische Schritt ist, diesen UI-Client direkt an den lokalen Neral-Endpunkt anzuschließen.`),360)}
function resetChat(){state.messages=[];chatThread.querySelectorAll('.message').forEach(x=>x.remove());chatEmpty.style.display='block';input.value='';input.style.height='auto';setMode('chat');setView('chat')}
function openRuntime(){runtimePanel.hidden=false;scrim.hidden=false}
function closeRuntime(){runtimePanel.hidden=true;scrim.hidden=true}

document.addEventListener('click',e=>{
  const view=e.target.closest('[data-view]');if(view){setView(view.dataset.view);return}
  const open=e.target.closest('[data-open]');if(open){setView(open.dataset.open);return}
  const mode=e.target.closest('[data-mode]');if(mode){setMode(mode.dataset.mode);return}
  const prompt=e.target.closest('[data-prompt]');if(prompt){setView('chat');setMode(prompt.dataset.suggestMode||'chat');input.value=prompt.dataset.prompt;input.dispatchEvent(new Event('input'));input.focus();return}
  const history=e.target.closest('[data-history]');if(history){setView('chat');showToast('Chat-Verlauf wird später an echte lokale Historie angebunden.');return}
  const action=e.target.closest('[data-action]');if(!action)return;
  const name=action.dataset.action;
  if(name==='new-chat')resetChat();
  if(name==='model')openRuntime();
  if(name==='close-model')closeRuntime();
  if(name==='settings')showToast('Einstellungen folgen als eigener App-Screen.');
  if(name==='attach')showToast('Datei- und Bildanhänge folgen mit der Runtime-Integration.');
  if(name==='voice')showToast('Spracheingabe ist noch nicht angebunden.');
  if(name==='more')showToast('Weitere Chat-Aktionen folgen mit echter Historie.');
  if(name==='collapse')showToast('Kompakte Sidebar folgt im nächsten UI-Schritt.');
});

scrim.addEventListener('click',closeRuntime);
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!runtimePanel.hidden)closeRuntime()});

document.getElementById('composerForm').addEventListener('submit',e=>{e.preventDefault();const text=input.value.trim();if(!text)return;addMessage('user',text);input.value='';input.style.height='auto';fakeHeliosReply(text)});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,170)+'px'});
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();document.getElementById('composerForm').requestSubmit()}});

if('serviceWorker' in navigator){window.addEventListener('load',()=>navigator.serviceWorker.register('./sw.js').catch(()=>{}))}
setMode('chat');setView('chat');
