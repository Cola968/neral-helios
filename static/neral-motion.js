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
    const mark=document.createElement("span");
    mark.className="neral-status-symbol";
    mark.setAttribute("aria-hidden","true");
    const description=document.createElement("span");
    description.className="neral-activity-label";
    description.textContent=label;
    row.append(mark,description);
    return row;
  }
  const preview=document.getElementById("messages");
  if(preview){
    let active=null,searches=0;
    const labels={searching:"Sucht im Web",checking:"Prüft Code",
      uploading:"Analysiert Anhänge"};
    window.addEventListener("neral:activity",event=>{
      const detail=event.detail||{};
      if(detail.phase==="done"||detail.phase==="cancelled"||
         detail.phase==="error"){
        active?.remove();active=null;searches=0;return;
      }
      if(detail.phase==="archive")return; // No synthetic timeline after a reply.
      if(detail.phase==="search-done"){
        if(active){
          searches=Number.isInteger(detail.searches)&&detail.searches>0
            ?Math.min(detail.searches,5):1;
          active.dataset.searches=String(searches);
        }
        return;
      }
      if(detail.phase==="files-done")return;
      if(!["thinking","searching","checking","uploading"].includes(detail.phase))return;
      const label=labels[detail.phase]||
        (detail.mode==="think"?"Denkt nach":"Verarbeitet Anfrage");
      if(!active){active=makeActivity(label,detail.phase);preview.append(active);}
      else{active.dataset.phase=detail.phase;
        active.querySelector(".neral-activity-label").textContent=label;}
      // Search-count annotation only follows a completed, actual web request.
      if(searches && detail.phase==="thinking" && !active.querySelector(".neral-search-count")){
        const note=document.createElement("span");
        note.className="neral-search-count";
        note.textContent=searches===1?"1 Suche ausgeführt":searches+" Suchen ausgeführt";
        active.append(note);
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
        const mark=document.createElement("span");
        mark.className="neral-presence neral-presence-pending";
        const signal=document.createElement("span");
        signal.className="neral-presence-signal";
        signal.setAttribute("aria-hidden","true");
        mark.append(image,signal);
        pending.prepend(mark);
      }
    };
    const observer=new MutationObserver(markPending);
    observer.observe(log,{childList:true});
    markPending();
    const stop=document.getElementById("stop");
    stop?.addEventListener("click",()=>{
      // The backend cancel may take longer; stop the visual activity now.
      for(const pending of log.querySelectorAll(".message.pending"))
        pending.classList.add("neral-cancel-requested");
    });
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
