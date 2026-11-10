"use strict";
/* NERAL Live Search: NDJSON events from actual contacted providers and
 * reachable HTTPS search hits. No page-reading or fabricated steps. */
(() => {
  async function search({query,csrf,signal,onEvent}){
    const options={method:"POST",credentials:"same-origin",cache:"no-store",
      headers:{"Content-Type":"application/json","X-Neral-CSRF":csrf},
      body:JSON.stringify({query,verify_links:true}),signal};
    let response=await fetch("/api/web/search/stream",options);
    if(response.status===404||response.status===405){
      // An older local installation still works until the entire ZIP updates.
      response=await fetch("/api/web/search",options);
      const payload=await response.json();
      if(!response.ok)throw new Error(payload.error||"Websuche nicht verfügbar.");
      return payload;
    }
    if(!response.ok){
      let message="Websuche nicht verfügbar.";
      try{message=(await response.json()).error||message;}catch{}
      throw new Error(message);
    }
    if(!response.body||typeof response.body.getReader!=="function")
      throw new Error("Der Browser unterstützt keine Live-Websuche.");
    const reader=response.body.getReader(),decoder=new TextDecoder();
    let buffer="",result=null,ended=false;
    const consume=line=>{
      if(!line.trim())return;
      let event;
      try{event=JSON.parse(line);}catch{throw new Error("Ungültiges Live-Suchergebnis.");}
      if(!event||typeof event!=="object")throw new Error("Ungültiges Live-Suchergebnis.");
      if(event.type==="error")throw new Error(String(event.message||"Websuche fehlgeschlagen."));
      if(event.type==="provider"&&typeof event.name==="string"){
        onEvent?.({type:"provider",name:event.name});
      }else if(event.type==="source"&&event.source?.reachable===true){
        onEvent?.({type:"source",source:event.source});
      }else if(event.type==="result"&&event.result&&typeof event.result==="object"){
        result=event.result;
      }else if(event.type==="end")ended=true;
    };
    try{
      while(true){
        const {done,value}=await reader.read();
        if(done)break;
        buffer+=decoder.decode(value,{stream:true});
        if(buffer.length>200000)throw new Error("Live-Suchantwort zu groß.");
        let end;
        while((end=buffer.indexOf("\n"))!==-1){
          const line=buffer.slice(0,end);buffer=buffer.slice(end+1);
          consume(line);
        }
      }
      buffer+=decoder.decode();
      if(buffer.trim())consume(buffer);
      if(!ended||!result)throw new Error("Live-Websuche wurde unterbrochen.");
      return result;
    }finally{reader.releaseLock?.();}
  }
  window.NeralWebLive={search};
})();
