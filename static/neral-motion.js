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
    // The approved NERAL symbol stays still. Only a small state indicator
    // moves while there is a real pending action.
    const mark=document.createElement("span");
    mark.className="neral-presence";
    const signal=document.createElement("span");
    signal.className="neral-presence-signal";
    signal.setAttribute("aria-hidden","true");
    mark.append(image,signal);
    row.append(mark,description);
    return row;
  }
  const preview=document.getElementById("messages");
  if(preview){
    let active=null,steps=[],completed=null;
    const phaseNames={
      thinking:"thinking",searching:"searching",checking:"checking",uploading:"uploading"
    };
    const activeLabel=event=>{
      const phase=event.detail?.phase;
      if(phase==="checking")return "Prüft Python-Syntax";
      if(phase==="searching")return "Durchsucht das Web";
      if(phase==="uploading")return "Verarbeitet ausgewählte Anhänge";
      return event.detail?.mode==="think"?"Denkt nach":"Verarbeitet Anfrage";
    };
    function step(label,key){
      const existing=steps.find(item=>item.key===key);
      if(existing)return existing;
      const item={key,label,state:"running"};
      steps.push(item);
      return item;
    }
    function finish(key,label){
      const entry=steps.find(item=>item.key===key);
      if(!entry)return;
      entry.state="done";
      if(label)entry.label=label;
    }
    function createHistory(className,summaryText){
      const details=document.createElement("details");
      details.className=className;
      const summary=document.createElement("summary");
      summary.textContent=summaryText;
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
      // Update the existing disclosure in place. Replacing an open details
      // element for every token/tool event makes it flicker and loses focus.
      if(existing && steps.length){
        const label=existing.querySelector("summary");
        const list=existing.querySelector(".neral-activity-history-list");
        if(label && list && typeof list.replaceChildren==="function"){
          label.textContent="Aktivitäten · "+steps.length;
          const entries=steps.map(entry=>{
            const li=document.createElement("li");
            li.dataset.state=entry.state;
            li.textContent=entry.label;
            return li;
          });
          list.replaceChildren(...entries);
          return;
        }
      }
      existing?.remove();
      if(!steps.length)return;
      const disclosure=createHistory("neral-activity-disclosure",
        "Aktivitäten · "+steps.length);
      if(keepOpen)disclosure.open=true;
      active.append(disclosure);
    }
    window.addEventListener("neral:activity",event=>{
      const phase=event.detail?.phase;
      // A completed record can only attach to its own immediately following
      // verified assistant response. Aborts, errors and clears discard it.
      if(phase==="archive"){
        if(completed?.length){
          const last=preview.lastElementChild;
          if(last?.classList?.contains("assistant")){
            steps=completed;
            last.append(createHistory("neral-activity-history",
              "Aktivitäten · "+steps.length+" Schritte"));
          }
        }
        completed=null;
        steps=[];
        return;
      }
      if(phase==="cancelled"||phase==="error"){
        active?.remove();active=null;steps=[];completed=null;
        return;
      }
      if(phase==="done"){
        for(const entry of steps)if(entry.state==="running")entry.state="done";
        completed=steps.length?steps.map(x=>({...x})):null;
        active?.remove();active=null;steps=[];
        return;
      }
      if(phase==="search-done"){
        const count=event.detail?.count;
        finish("web",Number.isInteger(count)&&count>0
          ?"Websuche abgeschlossen · "+Math.min(count,100)+" Treffer"
          :"Websuche abgeschlossen");
        renderActive();return;
      }
      if(phase==="files-done"){
        finish("files","Lokale Anhangsanfrage abgeschlossen");
        renderActive();return;
      }
      if(!phaseNames[phase])return;
      completed=null;
      const label=activeLabel(event);
      if(!active){
        active=makeActivity(label,phase);
        preview.append(active);
      }else{
        active.dataset.phase=phase;
        active.querySelector(".neral-activity-label").textContent=label;
      }
      if(phase==="searching")step("Websuche läuft","web");
      if(phase==="checking")step("Prüft Python-Syntax","codecheck");
      if(phase==="uploading"){
        const count=Number.isInteger(event.detail?.count)?
          Math.max(1,Math.min(3,event.detail.count)):null;
        step(count?"Lokale Anhangsanfrage · "+count+" Dateien":
          "Lokale Anhangsanfrage","files");
      }
      if(phase==="thinking"){
        step(event.detail?.mode==="think"?"NERAL HIGH verarbeitet Anfrage":
          "NERAL LOW verarbeitet Anfrage","model");
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
