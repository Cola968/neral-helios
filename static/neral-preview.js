"use strict";
// This client uses same-origin /api calls only. No persistent chat storage.
(() => {
  const $=id=>document.getElementById(id);
  const form=$("chatForm"),prompt=$("prompt"),submit=$("submit"),stop=$("stop"),errors=$("error"),messages=$("messages");
  const memory=[];
  let csrf=null,busy=false,inflight=null,conversationEpoch=0;
  stop.disabled=true;
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
    if(busy) {errors.textContent="NERAL berechnet noch die vorherige Anfrage.";return;}
    if(!csrf) {errors.textContent="Kein verifizierter lokaler Modellzugang.";return;}
    const text=prompt.value.trim();
    if(!text||encoder.encode(text).length>4096){errors.textContent="Nachricht zu lang oder leer.";return;}
    const mode=form.querySelector('input[name="mode"]:checked')?.value==="think" ? "think":"generate";
    busy=true;submit.disabled=true;stop.disabled=false;errors.textContent="";
    const controller=new AbortController();
    inflight=controller;
    const epochAtStart=conversationEpoch;
    // Manual browser cancellation and timeout are separate. Neither
    // can guarantee the already-running local Ollama GPU work stops.
    const timeout=AbortSignal.timeout(mode==="think"?190000:120000);
    const signal=AbortSignal.any([controller.signal,timeout]);
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
    try {
      const result=await json("/api/neral/chat/"+mode,{
        method:"POST",headers:{"Content-Type":"application/json","X-Neral-CSRF":csrf},
        body:JSON.stringify({prompt:text,history}),
        signal
      });
      // A late HTTP response must never resurrect a cleared chat.
      if(controller.signal.aborted||epochAtStart!==conversationEpoch)return;
      if(typeof result.text!=="string"||!result.text.trim())throw new Error("Leere Modellantwort.");
      if(result.model_origin!=="local_foundation_not_neral_adapter"||result.adapter_loaded!==false){
        throw new Error("Modellherkunft stimmt nicht mit der aktiven Sitzung überein.");
      }
      append("user",text);
      const verified=result.verified===true&&result.verification_scope==="numeric_final_only";
      const origin=result.model_generated===false&&result.tool_used ? "NERAL · lokal geprüft" :
                   verified ? "NERAL THINK · Endwert geprüft" :
                   mode==="think" ? "NERAL THINK · ungeprüft" : "NERAL FAST";
      append("assistant",result.text,origin);
      memory.push({role:"user",content:text},{role:"assistant",content:result.text});
      if(memory.length>12) memory.splice(0,memory.length-12);
      prompt.value="";
    } catch(e) {
      if(epochAtStart===conversationEpoch){
        errors.textContent=controller.signal.aborted
          ? "Browseranfrage abgebrochen. Ollama kann lokal noch weiterrechnen."
          : timeout.aborted
          ? "Zeitlimit der Browseranfrage erreicht. Ollama kann noch weiterrechnen."
          : e.message;
      }
    } finally {
      if(inflight===controller){
        inflight=null;busy=false;submit.disabled=false;stop.disabled=true;prompt.focus();
      }
    }
  });
  stop.addEventListener("click",()=>{
    if(inflight&&!inflight.signal.aborted){
      inflight.abort();
      errors.textContent="Browseranfrage abgebrochen. Ollama kann lokal noch weiterrechnen.";
    }
  });
  $("clear").addEventListener("click",()=>{
    conversationEpoch++;
    if(inflight&&!inflight.signal.aborted) inflight.abort();
    memory.length=0;messages.replaceChildren();
    append("assistant","Der Verlauf in diesem Fenster wurde gelöscht.","NERAL · lokal");
    errors.textContent="";prompt.focus();
  });
  connect();
})();
