"use strict";
/* NERAL motion V2: unobtrusive visual feedback tied to real browser/model state.
   Never fake percentages, token streaming or private reasoning traces. */
(() => {
  const iconUrl="/static/icon.svg";
  function favicon(host){
    const img=document.createElement("img");
    img.className="neral-live-site-icon";
    img.src="/api/web/favicon?host="+encodeURIComponent(host);
    img.alt="";
    img.setAttribute("alt","");
    img.setAttribute("aria-hidden","true");
    img.loading="lazy";
    img.decoding="async";
    img.referrerPolicy="no-referrer";
    img.addEventListener?.("load",()=>img.classList.add("neral-favicon-loaded"));
    img.addEventListener?.("error",()=>img.remove());
    return img;
  }
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
    const header=document.createElement("div");
    header.className="neral-status-header";
    header.append(mark,description);
    row.append(header);
    return row;
  }
  const preview=document.getElementById("messages");
  if(preview){
    let active=null,searches=0,startedAt=0,timer=null;
    const providers=new Set(),sourceHosts=new Set();
    function reset(){
      active?.remove();active=null;searches=0;startedAt=0;
      providers.clear();sourceHosts.clear();
      if(timer!==null && typeof clearInterval==="function")clearInterval(timer);
      timer=null;
    }
    function detailsArea(){
      if(!active)return null;
      let section=active.querySelector(".neral-live-sources");
      if(!section){
        section=document.createElement("div");
        section.className="neral-live-sources";
        active.append(section);
      }
      return section;
    }
    function showProvider(name){
      if(!active||active.dataset.phase!=="searching" ||
         !["DuckDuckGo HTML","Bing RSS"].includes(name) || providers.has(name))return;
      providers.add(name);
      const box=detailsArea();
      let line=box.querySelector(".neral-live-provider");
      if(!line){
        line=document.createElement("div");line.className="neral-live-provider";
        box.append(line);
      }
      line.textContent="Suchanbieter: "+[...providers]
        .map(n=>n==="DuckDuckGo HTML"?"DuckDuckGo":"Bing").join(" · ");
    }
    function showSearchQuery(query){
      if(!active||active.dataset.phase!=="searching"||
         typeof query!=="string"||!query.trim()||
         query.length>160||/[\r\n]/.test(query))return;
      const box=detailsArea();
      let line=box.querySelector(".neral-live-search-query");
      if(!line){
        line=document.createElement("div");
        line.className="neral-live-search-query";
        box.prepend?.(line);
        if(!line.parentNode&&!box.children?.includes?.(line))box.append(line);
      }
      line.textContent="Suchanfrage: "+query;
    }
    function showSource(source){
      if(!active || source?.reachable!==true || typeof source.url!=="string" ||
         sourceHosts.size>=5)return;
      let parsed;
      try{parsed=new URL(source.url);}catch{return;}
      const host=parsed.hostname.toLowerCase();
      if(parsed.protocol!=="https:"||parsed.username||parsed.password||
         parsed.port||!host.includes(".")||sourceHosts.has(host)||
         /^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$/.test(host)||
         /(?:^|\.)(?:localhost|local|internal|test|invalid)$/.test(host))return;
      sourceHosts.add(host);
      const box=detailsArea();
      let line=box.querySelector(".neral-live-sites");
      if(!line){
        line=document.createElement("div");line.className="neral-live-sites";
        const label=document.createElement("span");
        label.className="neral-live-sites-label";label.textContent="Gefundene Seiten:";
        line.append(label);box.append(line);
      }
      const link=document.createElement("a");
      link.className="neral-live-site";
      link.href=parsed.href;link.target="_blank";link.rel="noopener noreferrer";
      link.setAttribute("title",String(source.title||host).slice(0,130));
      link.textContent=host.replace(/^www\./,"");
      const thumb=document.createElement("span");
      thumb.className="neral-live-icon-wrap";
      thumb.setAttribute("aria-hidden","true");
      thumb.textContent=host.replace(/^www\./,"").charAt(0).toUpperCase();
      thumb.append(favicon(host));
      if(typeof link.prepend==="function")link.prepend(thumb);
      else link.append(thumb);
      line.append(link);
    }
    const labels={searching:"Sucht im Web",checking:"Prüft Code",
      uploading:"Analysiert Anhänge"};
    window.addEventListener("neral:activity",event=>{
      const detail=event.detail||{};
      if(detail.phase==="done"||detail.phase==="cancelled"||
         detail.phase==="error"){
        reset();return;
      }
      if(detail.phase==="search-query"){
        showSearchQuery(detail.query);return;
      }
      if(detail.phase==="search-provider"){
        showProvider(detail.name);return;
      }
      if(detail.phase==="source-found"){
        showSource(detail.source);return;
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
      if(!active){
        active=makeActivity(label,detail.phase);preview.append(active);
        startedAt=Date.now();
        if(typeof setInterval==="function"){
          timer=setInterval(()=>{
            if(!active)return;
            const seconds=Math.floor((Date.now()-startedAt)/1000);
            if(seconds<2)return;
            let clock=active.querySelector(".neral-activity-time");
            if(!clock){
              clock=document.createElement("span");
              clock.className="neral-activity-time";
              active.querySelector(".neral-status-header").append(clock);
            }
            clock.textContent="Arbeitet seit "+seconds+" s";
          },1000);
        }
      }else{active.dataset.phase=detail.phase;
        active.querySelector(".neral-activity-label").textContent=label;}
      // Search-count annotation only follows a completed, actual web request.
      if(searches && !providers.size && detail.phase==="thinking" && !active.querySelector(".neral-search-count")){
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
