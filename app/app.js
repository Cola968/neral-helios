const state={view:'home',messages:[]};
const titles={home:['Home','Dein lokaler KI-Arbeitsbereich'],chat:['Chat','Direkt mit Neral sprechen'],solve:['Solve','Mathe & Physik'],projects:['Projects','Chats, Dateien und Kontext']};
const views={home:document.getElementById('homeView'),chat:document.getElementById('chatView'),solve:document.getElementById('solveView'),projects:document.getElementById('projectsView')};
const title=document.getElementById('viewTitle');
const subtitle=document.getElementById('viewSubtitle');
const chatThread=document.getElementById('chatThread');
const chatEmpty=document.getElementById('chatEmpty');
const input=document.getElementById('composerInput');
const toast=document.getElementById('toast');

function showToast(text){toast.textContent=text;toast.classList.add('show');clearTimeout(showToast.t);showToast.t=setTimeout(()=>toast.classList.remove('show'),2200)}
function setView(name){if(!views[name])return;state.view=name;Object.entries(views).forEach(([k,v])=>v.classList.toggle('active',k===name));document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===name));title.textContent=titles[name][0];subtitle.textContent=titles[name][1];if(name==='chat')setTimeout(()=>input.focus(),50)}
function addMessage(role,text){chatEmpty.style.display='none';const el=document.createElement('div');el.className=`message ${role}`;el.textContent=text;chatThread.appendChild(el);state.messages.push({role,text});chatThread.scrollTop=chatThread.scrollHeight}
function fakeHeliosReply(prompt){setTimeout(()=>addMessage('assistant',`Runtime noch nicht verbunden. Dein Prompt „${prompt.slice(0,90)}${prompt.length>90?'…':''}“ ist im UI angekommen. Als Nächstes verbinden wir hier die lokale Neral-Inference.`),420)}

document.addEventListener('click',e=>{
  const view=e.target.closest('[data-view]');if(view){setView(view.dataset.view);return}
  const open=e.target.closest('[data-open]');if(open){setView(open.dataset.open);return}
  const prompt=e.target.closest('[data-prompt]');if(prompt){setView('chat');input.value=prompt.dataset.prompt;input.focus();return}
  const action=e.target.closest('[data-action]');if(action){
    if(action.dataset.action==='new-chat'){state.messages=[];chatThread.querySelectorAll('.message').forEach(x=>x.remove());chatEmpty.style.display='block';setView('chat')}
    if(action.dataset.action==='settings')showToast('Settings-Screen kommt als nächster App-Baustein.');
    if(action.dataset.action==='model')showToast('Neral 50M · lokaler Runtime-Status wird später live angebunden.');
    if(action.dataset.action==='resume'){setView('projects');showToast('Projekt „Neral / Helios“ geöffnet.');}
    if(action.dataset.action==='menu')showToast('Mobile Navigation ist unten verfügbar.');
  }
});

document.getElementById('composerForm').addEventListener('submit',e=>{e.preventDefault();const text=input.value.trim();if(!text)return;addMessage('user',text);input.value='';input.style.height='auto';fakeHeliosReply(text)});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,170)+'px'});
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('composerForm').requestSubmit()}});

if('serviceWorker' in navigator){window.addEventListener('load',()=>navigator.serviceWorker.register('./sw.js').catch(()=>{}))}
setView('home');
