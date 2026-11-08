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
    const dots=document.createElement("span");
    dots.className="neral-motion-dots";
    dots.setAttribute("aria-hidden","true");
    for(let i=0;i<3;i++)dots.append(document.createElement("span"));
    const description=document.createElement("span");
    description.className="neral-activity-label";
    description.textContent=label;
    row.append(dots,description);
    return row;
  }
  const preview=document.getElementById("messages");
  if(preview){
    let active=null,steps=[],completed=null;
    const phases=new Set(["thinking","searching","checking","uploading"]);
    function labelFor(detail){
      switch(detail.phase){
        case "searching": return "Sucht im Web";
        case "checking": return "Prüft Python-Syntax";
        case "uploading": return "Verarbeitet Anhänge";
        default: return detail.mode==="think"?"Denkt nach":"Verarbeitet Anfrage";
      }
    }
    function begin(key,label){
      const existing=steps.find(item=>item.key===key);
      if(existing)return;
      steps.push({key,label,state:"running"});
    }
    function finish(key,label){
      const entry=steps.find(item=>item.key===key);
      if(!entry)return;
      entry.state="done";
      if(label)entry.label=label;
    }
    function createHistory(className){
      const details=document.createElement("details");
      details.className=className;
      const summary=document.createElement("summary");
      summary.textContent="Verwendete Werkzeuge · "+steps.length;
      const list=document.createElement("ol");
      list.className="neral-activity-history-list";
      for(const entry of steps){
        const li=document.createElement("li");
        li.dataset.state=entry.state;
        li.textContent=entry.label;
        list.append(li);
      }
      details.append(summary,list);
      return details;
    }
    function renderActive(){
      if(!active)return;
      const existing=active.querySelector(".neral-activity-disclosure");
      const keepOpen=existing?.open===true;
      existing?.remove();
      // Model computation is not an observable tool step: no fake timeline.
      if(!steps.length)return;
      const disclosure=createHistory("neral-activity-disclosure");
      if(keepOpen)disclosure.open=true;
      active.append(disclosure);
    }
    window.addEventListener("neral:activity",event=>{
      const detail=event.detail||{},phase=detail.phase;
      if(phase==="archive"){
        if(completed?.length){
          const last=preview.lastElementChild;
          if(last?.classList?.contains("assistant")){
            steps=completed;
            last.append(createHistory("neral-activity-history"));
          }
        }
        completed=null;steps=[];return;
      }
      if(phase==="cancelled"||phase==="error"){
        active?.remove();active=null;steps=[];completed=null;return;
      }
      if(phase==="done"){
        // Only a successful response may archive; the caller emits archive.
        for(const entry of steps)if(entry.state==="running")entry.state="done";
        completed=steps.length?steps.map(entry=>({...entry})):null;
        active?.remove();active=null;steps=[];return;
      }
      if(phase==="search-done"){
        const count=detail.count;
        finish("web",Number.isInteger(count)&&count>0
          ?"Websuche · "+Math.min(count,100)+" relevante Treffer"
          :"Websuche · keine relevanten Treffer");
        renderActive();return;
      }
      if(phase==="files-done"){
        finish("files","Anhänge verarbeitet");
        renderActive();return;
      }
      if(!phases.has(phase))return;
      completed=null;
      const label=labelFor(detail);
      if(!active){
        active=makeActivity(label,phase);
        preview.append(active);
      }else{
        active.dataset.phase=phase;
        active.querySelector(".neral-activity-label").textContent=label;
      }
      // Record only real operations, never synthetic thinking phases.
      if(phase==="searching")begin("web","Websuche läuft");
      if(phase==="checking")begin("codecheck","Prüft Python-Syntax");
      if(phase==="uploading"){
        const count=Number.isInteger(detail.count) && detail.count>0
          ?Math.min(detail.count,100):null;
        begin("files",count?"Anhangsanfrage · "+count+" Dateien":"Anhangsanfrage");
      }
      renderActive();
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
