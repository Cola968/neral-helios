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
  const api={read};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(root)root.NeralLiveStream=api;
})(typeof window!=="undefined"?window:undefined);
