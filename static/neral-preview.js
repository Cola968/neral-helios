"use strict";
// This client uses same-origin /api calls only. No persistent chat storage.
(() => {
  const $=id=>document.getElementById(id);
  const form=$("chatForm"),prompt=$("prompt"),submit=$("submit"),errors=$("error"),messages=$("messages");
  const memory=[];
  let csrf=null,busy=false;
  const encoder=new TextEncoder();

  function append(role,text,origin) {
    const empty=messages.querySelector(".welcome");
    if(empty) empty.remove();
    const item=document.createElement("div");
    item.className="message "+role;
    const heading=document.createElement("strong");
    heading.textContent=role==="user" ? "Du" : (origin||"NERAL");
    const body=document.createElement("p");
    body.textContent=text; // Never evaluate/render model HTML, scripts, Markdown.
    item.append(heading,body);
    messages.append(item);
    messages.scrollTop=messages.scrollHeight;
  }
  function showSources(sources) {
    if(!Array.isArray(sources)) return;
    for(const source of sources.slice(0,3)){
      if(!source || typeof source.url!=="string" ||
         !source.url.startsWith("https://") ||
         typeof source.title!=="string") continue;
      const line=document.createElement("div");
      line.className="web-source";
      const link=document.createElement("a");
      link.href=source.url;
      link.target="_blank";
      link.rel="noopener noreferrer";
      link.textContent="["+String(source.id)+"] "+source.title;
      line.append(link);
      messages.append(line);
    }
  }
  async function json(path, init={}) {
    const response=await fetch(path,{credentials:"same-origin",cache:"no-store",...init});
    const data=await response.json();
    if(!response.ok) throw new Error(data.error||"Lokale NERAL-Anfrage fehlgeschlagen.");
    return data;
  }
  async function connect() {
    try {
      const session=await json("/api/neral/preview/session");
      csrf=session.csrf;
      const status=await json("/api/neral/chat/status");
      if(status.model_origin!=="local_foundation_not_neral_adapter" ||
         status.model!==session.model || status.adapter_loaded!==false) {
        throw new Error("Unbekannte Modellherkunft. NERAL-Chat verweigert.");
      }
      $("model").textContent=status.model;
      $("connection").textContent="Lokal verbunden · "+status.model;
    } catch(e) {
      csrf=null;
      $("connection").textContent="Nicht verbunden";
      errors.textContent=e.message;
    }
  }
  form.addEventListener("submit",async event=>{
    event.preventDefault();
    if(busy || !csrf) {errors.textContent="Kein verifizierter lokaler Modellzugang.";return;}
    const text=prompt.value.trim();
    if(!text||encoder.encode(text).length>4096){errors.textContent="Nachricht zu lang oder leer.";return;}
    const mode=form.querySelector('input[name="mode"]:checked')?.value==="think" ? "think":"generate";
    const useWeb=$("webSearch").checked===true;
    const webQuery=$("webQuery").value.trim();
    if(useWeb && (!webQuery || webQuery.length<2 || webQuery.length>160 ||
                  /[\\x00-\\x1f\\x7f]/.test(webQuery))){
      errors.textContent="Websuche: Suchbegriff mit 2–160 normalen Zeichen erforderlich.";
      return;
    }
    busy=true;submit.disabled=true;errors.textContent="";
    // Preserve user/assistant pairs while keeping every sent item under
    // the server's 2048-byte limit and total history below 12 KB.
    // Keep the tail of long answers, where conclusions normally appear.
    let history=memory.slice(-12).map(m=>{
      let content=m.content;
      while(encoder.encode(content).length>1900){
        content=content.slice(Math.max(1,Math.floor(content.length/8)));
      }
      return {role:m.role,content};
    });
    while(history.length && encoder.encode(JSON.stringify(history)).length>9000){
      history=history.slice(2);
    }
    const path=useWeb?"/api/neral/preview/web-answer":"/api/neral/chat/"+mode;
    const payload=useWeb ? {prompt:text,history,query:webQuery,
                            consent:true,mode:mode==="think"?"think":"fast"}
                         : {prompt:text,history};
    try {
      const result=await json(path,{
        method:"POST",headers:{"Content-Type":"application/json","X-Neral-CSRF":csrf},
        body:JSON.stringify(payload),
        signal:AbortSignal.timeout(mode==="think"?190000:120000)
      });
      if(typeof result.text!=="string"||!result.text.trim())throw new Error("Leere Modellantwort.");
      if(result.model_origin!=="local_foundation_not_neral_adapter"||result.adapter_loaded!==false){
        throw new Error("Modellherkunft stimmt nicht mit der aktiven Sitzung überein.");
      }
      if(useWeb && (result.web_search_used!==true ||
                    result.source_claims_verified!==false ||
                    !Array.isArray(result.sources))) {
        throw new Error("Unvollständige Angaben zur Websuche.");
      }
      append("user",text);
      const verified=result.verified===true&&result.verification_scope==="numeric_final_only";
      const origin=useWeb ? "NERAL · Webtreffer ungeprüft" :
                   result.model_generated===false&&result.tool_used ? "NERAL · lokal geprüft" :
                   verified ? "NERAL THINK · Endwert geprüft" :
                   mode==="think" ? "NERAL THINK · ungeprüft" : "NERAL FAST";
      append("assistant",result.text,origin);
      if(useWeb) showSources(result.sources);
      memory.push({role:"user",content:text},{role:"assistant",content:result.text});
      if(memory.length>12) memory.splice(0,memory.length-12);
      prompt.value="";
      if(useWeb){$("webSearch").checked=false;$("webQuery").value="";}
    } catch(e) {errors.textContent=e.name==="TimeoutError"?"Lokales Modell hat das Zeitlimit überschritten.":e.message;}
    finally {busy=false;submit.disabled=false;prompt.focus();}
  });
  $("clear").addEventListener("click",()=>{memory.length=0;messages.replaceChildren();append("assistant","Der Verlauf in diesem Fenster wurde gelöscht.","NERAL · lokal");errors.textContent="";prompt.focus();});
  connect();
})();
