"use strict";
// NERAL 1.0 chat chrome: progressive enhancement only.
// Model, account, files, web and voice logic remain owned by their existing modules.
(() => {
  const byId=id=>document.getElementById(id);
  const sidebar=byId("neralSidebar");
  const toggle=byId("sidebarToggle");
  const closeButton=byId("sidebarClose");
  const backdrop=byId("sidebarBackdrop");
  const messages=byId("messages");
  const prompt=byId("prompt");
  const form=byId("chatForm");
  const submit=byId("submit");
  const toolMenu=byId("extraTools");
  const modelMenu=byId("modelDetails");
  if(!sidebar||!toggle||!backdrop||!prompt||!form||!submit)return;

  const narrow=window.matchMedia?.("(max-width: 780px)");
  let opened=false;
  const isMobile=()=>narrow?.matches===true;
  function setSidebar(next,{restoreFocus=false}={}){
    opened=!!next && isMobile();
    sidebar.classList.toggle("is-open",opened);
    backdrop.classList.toggle("is-open",opened);
    toggle.setAttribute("aria-expanded",opened?"true":"false");
    toggle.setAttribute("aria-label",opened?"Menü schließen":"Menü öffnen");
    sidebar.inert=isMobile()&&!opened;
    if(opened){
      closeButton?.focus();
    } else if(restoreFocus){
      toggle.focus();
    }
  }
  setSidebar(false);
  toggle.addEventListener("click",()=>setSidebar(!opened,{restoreFocus:opened}));
  closeButton?.addEventListener("click",()=>setSidebar(false,{restoreFocus:true}));
  backdrop.addEventListener("click",()=>setSidebar(false,{restoreFocus:true}));
  byId("savedChatList")?.addEventListener("click",event=>{
    if(event.target.closest(".saved-chat-open"))setSidebar(false);
  });
  byId("newSavedChat")?.addEventListener("click",()=>setSidebar(false));
  narrow?.addEventListener?.("change",()=>setSidebar(false));
  document.addEventListener("keydown",event=>{
    if(event.key!=="Escape")return;
    if(opened){setSidebar(false,{restoreFocus:true});return;}
    if(toolMenu?.open){toolMenu.open=false;return;}
    if(modelMenu?.open)modelMenu.open=false;
  });
  document.addEventListener("pointerdown",event=>{
    if(toolMenu?.open&&!toolMenu.contains(event.target))toolMenu.open=false;
    if(modelMenu?.open&&!modelMenu.contains(event.target))modelMenu.open=false;
  });

  // Use the real form submit path, with IME and multiline editing protected.
  function resizePrompt(){
    prompt.style.height="auto";
    prompt.style.height=Math.min(prompt.scrollHeight,200)+"px";
  }
  prompt.addEventListener("input",resizePrompt);
  prompt.addEventListener("keydown",event=>{
    if(event.key!=="Enter"||event.shiftKey||event.isComposing||
       event.ctrlKey||event.metaKey||event.altKey)return;
    event.preventDefault();
    if(!submit.disabled)form.requestSubmit(submit);
  });
  resizePrompt();
  for(const example of document.querySelectorAll("[data-prompt]")){
    example.addEventListener("click",()=>{
      if(!example.dataset.prompt)return;
      prompt.value=example.dataset.prompt;
      resizePrompt();
      prompt.focus();
    });
  }

  // Live regions remain unobtrusive until something actually needs attention.
  // Files rejected by validation and denied microphones must never be invisible.
  function watchStatusWarning(id,pattern){
    const node=byId(id);
    if(!node)return;
    const reflect=()=>{
      node.classList.toggle("neral-status-alert",
        pattern.test(node.textContent||""));
    };
    if(typeof MutationObserver!=="undefined"){
      const observer=new MutationObserver(reflect);
      observer.observe(node,{childList:true,characterData:true,subtree:true});
    }
    reflect();
  }
  watchStatusWarning("fileStatus",/abgelehnt|ungültig|fehler|zu groß|unzulässig/i);
  watchStatusWarning("voiceStatus",/fehlgeschlagen|verweigert|nicht verfügbar|konnte nicht|keine sprache|unterbrochen/i);
  watchStatusWarning("localVoiceStatus",/fehlgeschlagen|verweigert|nicht verfügbar|konnte|fehlen|nicht installiert|fehler/i);

  // A restored chat should scroll from its own already-rendered content.
  window.addEventListener("neral:load-chat",()=>{
    if(messages)messages.scrollTop=messages.scrollHeight;
    resizePrompt();
    setSidebar(false);
  });
})();
