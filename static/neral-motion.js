"use strict";
/* Small progressive enhancement for the existing Helios and NERAL Local UI.
   No network calls, synthetic token streaming, timers, or private reasoning display. */
(() => {
  const iconUrl="/static/icon.svg";
  function makeActivity(label,phase) {
    const row=document.createElement("div");
    row.className="neral-activity";
    row.setAttribute("role","status");
    row.setAttribute("aria-live","polite");
    row.dataset.phase=phase;
    const icon=document.createElement("img");
    icon.className="neral-motion-glyph";
    icon.src=iconUrl;
    icon.alt="";
    icon.setAttribute("aria-hidden","true");
    const text=document.createElement("span");
    text.className="neral-activity-label";
    text.textContent=label;
    row.append(icon,text);
    return row;
  }
  // Explicit lifecycle events from the preview's real async request.
  const preview=document.getElementById("messages");
  let previewActivity=null;
  if(preview){
    window.addEventListener("neral:activity",event=>{
      const phase=event.detail?.phase;
      if(phase==="done" || phase==="error"){
        previewActivity?.remove(); previewActivity=null; return;
      }
      if(phase!=="thinking" && phase!=="searching")return;
      const label=phase==="searching"?"Durchsucht das Web":event.detail?.mode==="think"?"Denkt nach":"Verarbeitet Anfrage";
      if(!previewActivity){
        previewActivity=makeActivity(label,phase);
        preview.append(previewActivity);
      }else{
        previewActivity.dataset.phase=phase;
        previewActivity.querySelector(".neral-activity-label").textContent=label;
      }
      preview.scrollTop=preview.scrollHeight;
    });
  }
  // Existing Helios chat has a real pending article. Attach a calm marker
  // only until the first actual partial response is available.
  const log=document.getElementById("log");
  if(log){
    const markPending=()=>{
      for(const pending of log.querySelectorAll(".message.pending")){
        if(pending.querySelector(".neral-pending-glyph"))continue;
        const icon=document.createElement("img");
        icon.className="neral-pending-glyph";
        icon.src=iconUrl;
        icon.alt="";
        icon.setAttribute("aria-hidden","true");
        pending.prepend(icon);
      }
    };
    new MutationObserver(markPending).observe(log,{childList:true});
    markPending();
  }
  // The separate local Qwen chat uses aria-busy on its transcript.
  // Unlike a guessed animation duration, this tracks the actual request.
  const transcript=document.getElementById("neralLocalChatAnswer");
  if(transcript){
    let localActivity=null;
    const reflectBusy=()=>{
      const busy=transcript.getAttribute("aria-busy")==="true";
      if(!busy){localActivity?.remove();localActivity=null;return;}
      if(localActivity)return;
      const thinking=document.getElementById("neralLocalChatThink")?.checked;
      localActivity=makeActivity(thinking?"Denkt nach":"Verarbeitet Anfrage","thinking");
      transcript.insertAdjacentElement("afterend",localActivity);
    };
    new MutationObserver(reflectBusy).observe(transcript,{attributes:true,attributeFilter:["aria-busy"]});
    reflectBusy();
  }
})();
