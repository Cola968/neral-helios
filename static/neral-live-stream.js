"use strict";
/* Genuine NDJSON from local NERAL/Ollama only: never invent typing tokens.
 * Final result is authoritative; incomplete streams may not enter history. */
(function(root){
  async function read(response,{onDelta=()=>{}}={}){
    if(!response?.ok){
      let error="Lokale Anfrage fehlgeschlagen.";
      try{
        const data=await response.json();
        if(typeof data?.error==="string"&&data.error.length<=180)error=data.error;
      }catch{}
      throw new Error(error);
    }
    const type=response.headers?.get?.("Content-Type")||response.headers?.get?.("content-type")||"";
    if(!type.toLowerCase().startsWith("application/x-ndjson"))
      throw new Error("Antwort ist kein NERAL-Datenstrom.");
    if(!response.body?.getReader)
      throw new Error("Dieser Browser unterstützt keinen Datenstrom.");
    const reader=response.body.getReader();
    const decoder=new TextDecoder("utf-8",{fatal:true});
    let buffer="",total=0,count=0,complete=null,streamed="",state="reading";
    function processLine(line){
      if(!line.trim())return;
      if(++count>6000||line.length>120000)throw new Error("Datenstrom zu groß.");
      const event=JSON.parse(line);
      if(!event||typeof event!=="object"||Array.isArray(event))
        throw new Error("Ungültiger Datenstrom.");
      if(state==="complete")throw new Error("Zusätzliche Daten nach Abschluss.");
      if(event.type==="delta"){
        const part=event.text;
        if(typeof part!=="string"||part.length>8192)
          throw new Error("Ungültiger Textabschnitt.");
        streamed+=part;
        if(new TextEncoder().encode(streamed).length>32768)
          throw new Error("Antwort zu groß.");
        onDelta(part);
      }else if(event.type==="final"){
        if(!event.result||typeof event.result!=="object"||
           typeof event.result.text!=="string"||!event.result.text.trim()||
           new TextEncoder().encode(event.result.text).length>32768)
          throw new Error("Unvollständige Modellantwort.");
        if(streamed && streamed!==event.result.text)
          throw new Error("Datenstrom und Antwort stimmen nicht überein.");
        complete=event.result;state="complete";
      }else if(event.type==="error"){
        throw new Error(typeof event.error==="string"&&event.error.length<=180
          ?event.error:"Lokale Modellgenerierung fehlgeschlagen.");
      }else throw new Error("Unbekannter Datenstrom-Eintrag.");
    }
    try{
      while(true){
        const {done,value}=await reader.read();
        if(done)break;
        if(!(value instanceof Uint8Array))
          throw new Error("Ungültiger Antwortdatenblock.");
        total+=value.byteLength;
        if(total>256*1024)throw new Error("Datenstrom zu groß.");
        buffer+=decoder.decode(value,{stream:true});
        if(buffer.length>120000)throw new Error("Datenstrom zu groß.");
        let index;
        while((index=buffer.indexOf("\n"))!==-1){
          const line=buffer.slice(0,index).replace(/\r$/,"");
          buffer=buffer.slice(index+1);
          processLine(line);
        }
      }
      buffer+=decoder.decode();
      if(buffer.trim())processLine(buffer.trim());
      if(state!=="complete"||!complete)
        throw new Error("Modellantwort nicht vollständig abgeschlossen.");
      return complete;
    }finally{
      reader.releaseLock?.();
    }
  }
  function createPainter(target,options={}){
    // Visual pacing is separate from inference: LOW uses actual incoming
    // Ollama text, HIGH only starts after its final checked answer exists.
    // No synthetic model tokens, no artificial progress percentages.
    const schedule=options.schedule||((fn)=>requestAnimationFrame(fn));
    const cancelFrame=options.cancelFrame||((id)=>cancelAnimationFrame(id));
    const current=options.isCurrent||(()=>true);
    const signal=options.signal;
    const reduced=()=>options.reducedMotion===true ||
      (options.reducedMotion!==false &&
       typeof matchMedia==="function"&&matchMedia("(prefers-reduced-motion: reduce)").matches);
    const hidden=()=>options.isHidden?.() ||
      (typeof document!=="undefined"&&document.hidden===true);
    let pending=[],offset=0,scheduled=null,finalRequested=false;
    let stopped=false,completed=false,resolveFinish=null,finished=null;

    function finishPromise(result){
      if(completed)return;
      completed=true;
      if(scheduled!==null){cancelFrame(scheduled);scheduled=null;}
      if(resolveFinish){resolveFinish(result);resolveFinish=null;}
    }
    function valid(){
      if(stopped||signal?.aborted||!current()){
        stopped=true;finishPromise(false);return false;
      }
      return true;
    }
    function paintAll(){
      if(!valid())return;
      if(offset<pending.length){
        target.textContent+=pending.slice(offset).join("");
        offset=pending.length;
        options.onPaint?.(target.textContent);
      }
      if(finalRequested)finishPromise(true);
    }
    function frame(){
      scheduled=null;
      if(!valid())return;
      if(reduced()||hidden()){paintAll();return;}
      const backlog=pending.length-offset;
      if(backlog){
        // One frame should show a fragment, not an entire 100-word block.
        // Increase pace under backlog to avoid delaying long answers.
        const amount=Math.min(backlog,96,Math.max(3,Math.ceil(backlog/90)));
        target.textContent+=pending.slice(offset,offset+amount).join("");
        offset+=amount;
        options.onPaint?.(target.textContent);
      }
      if(finalRequested&&offset===pending.length){finishPromise(true);return;}
      if(offset<pending.length)scheduled=schedule(frame);
    }
    function kick(){
      if(!valid())return;
      if(reduced()||hidden()){paintAll();return;}
      if(scheduled===null&&offset<pending.length)scheduled=schedule(frame);
      if(finalRequested&&offset===pending.length)finishPromise(true);
    }
    function push(text){
      if(completed||finalRequested||!valid()||typeof text!=="string"||!text)return;
      const pieces=typeof Intl!=="undefined"&&Intl.Segmenter?
        [...new Intl.Segmenter("de",{granularity:"grapheme"}).segment(text)].map(x=>x.segment):
        Array.from(text);
      pending.push(...pieces);
      kick();
    }
    function finish(){
      if(completed)return Promise.resolve(!stopped);
      if(finished)return finished;
      if(!valid())return Promise.resolve(false);
      finalRequested=true;
      finished=new Promise(resolve=>{resolveFinish=resolve;});
      kick();
      return finished;
    }
    function cancel(){
      stopped=true;finishPromise(false);
    }
    if(signal?.addEventListener)signal.addEventListener("abort",cancel,{once:true});
    return {push,finish,cancel};
  }
  const api={read,createPainter};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(root)root.NeralLiveStream=api;
})(typeof window!=="undefined"?window:undefined);
