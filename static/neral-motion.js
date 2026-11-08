"use strict";
/* NERAL motion V2: unobtrusive visual feedback tied to real browser/model state.
   Never fake percentages, token streaming or private reasoning traces. */
(() => {
  const iconUrl="/static/icon.svg";
  function makeActivity(label,phase){
    const row=document.createElement("div");
    row.className="neral-activity";
    row.dataset.phase=phase;
    row.setAttribute("role","status");
    row.setAttribute("aria-live","polite");
    const image=document.createElement("img");
    image.className="neral-motion-glyph";
    image.src=iconUrl;
    image.alt="";
    image.setAttribute("aria-hidden","true");
    const description=document.createElement("span");
    description.className="neral-activity-label";
    description.textContent=label;
    row.append(image,description);
    return row;
  }
  const preview=document.getElementById("messages");
  if(preview){
    let active=null;
    window.addEventListener("neral:activity",event=>{
      const phase=event.detail?.phase;
      if(phase==="done"||phase==="error"||phase==="cancelled"){
        active?.remove();
        active=null;
        return;
      }
      if(phase!=="thinking"&&phase!=="searching")return;
      const label=phase==="searching"?"Durchsucht das Web":
        event.detail?.mode==="think"?"Denkt nach":"Verarbeitet Anfrage";
      if(!active){
        active=makeActivity(label,phase);
        preview.append(active);
      }else{
        active.dataset.phase=phase;
        active.querySelector(".neral-activity-label").textContent=label;
      }
      preview.scrollTop=preview.scrollHeight;
    });
  }

  // Helios uses real incremental responses. Keep its already-rendered text in
  // place; a CSS caret is the only animated element after the first real token.
  const log=document.getElementById("log");
  if(log){
    const markPending=()=>{
      for(const pending of log.querySelectorAll(".message.pending")){
        if(pending.dataset.partial||pending.querySelector(".neral-pending-glyph"))continue;
        const image=document.createElement("img");
        image.className="neral-pending-glyph";
        image.src=iconUrl;
        image.alt="";
        image.setAttribute("aria-hidden","true");
        pending.prepend(image);
      }
    };
    const observer=new MutationObserver(markPending);
    observer.observe(log,{childList:true});
    markPending();
  }

  // Embedded local chat has an actual aria-busy lifecycle. No guessed timer.
  const transcript=document.getElementById("neralLocalChatAnswer");
  if(transcript){
    let active=null;
    const reflectBusy=()=>{
      const working=transcript.getAttribute("aria-busy")==="true";
      if(!working){
        active?.remove();
        active=null;
        return;
      }
      if(active)return;
      const isThink=document.getElementById("neralLocalChatThink")?.checked===true;
      active=makeActivity(isThink?"Denkt nach":"Verarbeitet Anfrage","thinking");
      transcript.insertAdjacentElement("afterend",active);
    };
    const observer=new MutationObserver(reflectBusy);
    observer.observe(transcript,{attributes:true,attributeFilter:["aria-busy"]});
    reflectBusy();
  }
})();
