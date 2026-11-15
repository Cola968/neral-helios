"use strict";
/* One verified web-search pipeline for the legacy Helios panel and the modern
 * NERAL chat. Browser requests are opt-in and contain only the explicit query.
 * Search snippets are untrusted; links and favicons stay on safe HTTPS hosts. */
(() => {
  const el=id=>document.getElementById(id);
  if(!el("webSearchToggle")||!el("webSearchPanel"))return;
  const panel=el("webSearchPanel"),toggle=el("webSearchToggle");
  const form=el("webSearchForm"),query=el("webSearchQuery");
  const results=el("webSearchResults"),status=el("webSearchStatus");
  if(!form||!query||!results||!status)return;
  let busy=false;
  const shown=new Map(),providers=new Set();

  toggle.addEventListener("click",()=>{
    panel.hidden=!panel.hidden;
    toggle.setAttribute("aria-expanded",String(!panel.hidden));
    if(!panel.hidden)query.focus();
  });
  el("webSearchClose")?.addEventListener("click",()=>{
    panel.hidden=true;
    toggle.setAttribute("aria-expanded","false");
    toggle.focus();
  });

  function isPublicHttps(address){
    try{
      const url=new URL(address);
      if(url.protocol!=="https:"||url.username||url.password||
         (url.port&&url.port!=="443"))return false;
      const host=url.hostname.toLowerCase();
      if(!/^[a-z0-9.-]+\.[a-z]{2,}$/i.test(host)||
         /(?:^|\.)(?:localhost|local|internal|test|invalid)$/i.test(host)||
         /(?:^|\.)(?:127\.0\.0\.1|0\.0\.0\.0)$/i.test(host)||
         /^(?:\d{1,3}\.){3}\d{1,3}$/.test(host))return false;
      return true;
    }catch{return false;}
  }

  function siteIcon(host){
    const icon=document.createElement("img");
    icon.className="neral-web-favicon";
    icon.alt="";
    icon.loading="eager";
    icon.decoding="async";
    icon.referrerPolicy="no-referrer";
    icon.setAttribute("aria-hidden","true");
    icon.addEventListener("load",()=>icon.classList.add("neral-favicon-loaded"));
    icon.addEventListener("error",()=>icon.remove());
    icon.src="/api/web/favicon?host="+encodeURIComponent(host);
    if(icon.complete&&icon.naturalWidth>0)icon.classList.add("neral-favicon-loaded");
    return icon;
  }

  function appendSource(item){
    if(!item||typeof item.url!=="string"||
       (item.reachable!==true&&item.discovered!==true)||
       !isPublicHttps(item.url))return;
    const uri=new URL(item.url),key=uri.href,host=uri.hostname.toLowerCase();
    const existing=shown.get(key);
    if(existing){
      if(item.reachable===true)existing.dataset.reachable="true";
      return;
    }
    if(shown.size>=5)return;
    const article=document.createElement("article");
    article.className="web-result";
    article.dataset.reachable=String(item.reachable===true);
    const heading=document.createElement("div");
    heading.className="web-result-head";
    const thumb=document.createElement("span");
    thumb.className="web-result-icon";
    thumb.setAttribute("aria-hidden","true");
    thumb.textContent=host.replace(/^www\./,"").charAt(0).toUpperCase();
    thumb.append(siteIcon(host));
    const meta=document.createElement("div");
    meta.className="web-result-meta";
    const link=document.createElement("a");
    link.href=uri.href;
    link.target="_blank";
    link.rel="noopener noreferrer";
    link.textContent=String(item.title||host).slice(0,180);
    const domain=document.createElement("small");
    domain.textContent=host.replace(/^www\./,"");
    meta.append(link,domain);
    heading.append(thumb,meta);
    article.append(heading);
    const snippet=String(item.snippet||"").trim().slice(0,350);
    if(snippet){
      const paragraph=document.createElement("p");
      paragraph.textContent=snippet;
      article.append(paragraph);
    }
    shown.set(key,article);
    results.append(article);
  }

  function updateStatus(done=false){
    const count=shown.size;
    if(count){
      const verified=[...shown.values()].filter(item=>item.dataset.reachable==="true").length;
      status.textContent=count+" passende Suchmaschinen-Treffer · "+
        verified+" HTTPS-Adressen geprüft · Seiteninhalte nicht gelesen"+
        (done?".":" · weitere Ergebnisse werden gesucht …");
      return;
    }
    if(done){
      status.textContent="Websuche abgeschlossen · keine passenden Treffer. Suchanbieter können die Ergebnisse einschränken.";
      return;
    }
    status.textContent="Sucht im Web"+(providers.size?" über "+[...providers].join(" · "):"")+" …";
  }

  form.addEventListener("submit",async event=>{
    event.preventDefault();
    if(busy)return;
    const text=query.value.trim();
    if(text.length<2||text.length>160||/[\r\n\x00-\x1f\x7f]/.test(text)){
      status.textContent="Suchbegriff muss 2–160 gültige Zeichen enthalten.";
      return;
    }
    busy=true;
    el("webSearchSubmit").disabled=true;
    shown.clear();providers.clear();results.replaceChildren();updateStatus();
    const signal=AbortSignal.timeout(45000);
    try{
      const onEvent=event=>{
        if(event.type==="provider"&&typeof event.name==="string"){
          providers.add(event.name);
          updateStatus();
        }else if(["source","candidate"].includes(event.type)&&event.source){
          appendSource(event.source);
          updateStatus();
        }
      };
      let payload;
      if(typeof window.NeralWebLive?.search==="function"){
        // The legacy app's session is cookie-protected. The modern local preview
        // sends its own CSRF token; the shared API still validates the query.
        payload=await window.NeralWebLive.search({query:text,csrf:"",signal,onEvent});
      }else{
        const response=await fetch("/api/web/search",{
          method:"POST",credentials:"same-origin",cache:"no-store",
          headers:{"Content-Type":"application/json"},
          body:JSON.stringify({query:text,verify_links:true}),signal
        });
        payload=await response.json();
        if(!response.ok)throw new Error(payload.error||"Websuche fehlgeschlagen.");
      }
      for(const hit of (Array.isArray(payload.results)?payload.results:[]))appendSource(hit);
      for(const hit of (Array.isArray(payload.previews)?payload.previews:[]))appendSource(hit);
      updateStatus(true);
    }catch(error){
      const message=error?.name==="TimeoutError"
        ?"Zeitüberschreitung der Websuche. Bitte Verbindung oder Suchanbieter prüfen."
        :String(error?.message||"Websuche nicht erreichbar.");
      status.textContent=message+(shown.size?" · Bereits gefundene Suchtreffer bleiben sichtbar.":"");
    }finally{
      busy=false;
      el("webSearchSubmit").disabled=false;
    }
  });
})();
