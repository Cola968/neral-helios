"use strict";
// Opt-in half-duplex, LOCAL CPU voice. Distinct from browser SpeechRecognition.
// No browser STT, speechSynthesis, remote endpoints or automatic model fetch.
(() => {
  const $=id=>document.getElementById(id);
  const button=$("localVoiceStart"), status=$("localVoiceStatus");
  if(!button||!status)return;
  const form=$("chatForm"), prompt=$("prompt");
  const browser=window;
  let active=false,starting=false,version=0,phase="idle",csrf=null;
  let stream=null,context=null,node=null,source=null,replyAudio=null,controller=null,cancelPlayback=null;
  let preRoll=[],blocks=[],heard=0,silence=0,seconds=0,awaiting=false;
  const SAMPLE_RATE=16000, MAX_SPEECH=12, END_SILENCE=.8;

  function say(text){status.textContent=text;}
  function setVoiceButtonLabel(text){
    // Keep the icon during responsive voice state changes.
    const label=typeof button.querySelector==="function"
      ? button.querySelector(".voice-label") : null;
    if(label)label.textContent=text;
    else button.textContent=text; // Test doubles have no DOM children.
    button.setAttribute("aria-label",text);
  }
  function resetVAD(){
    preRoll=[];blocks=[];heard=0;silence=0;seconds=0;
  }
  function turnOff(){
    version++;
    active=false;starting=false;phase="idle";awaiting=false;
    if(controller){controller.abort();controller=null;}
    // A stopped Audio element does not fire "ended". Resolve the pending
    // playback promise and revoke its blob URL before releasing the microphone.
    if(cancelPlayback)cancelPlayback();
    if(replyAudio){
      try{replyAudio.pause();replyAudio.currentTime=0;}catch{}
      replyAudio=null;
    }
    try{if(node)node.disconnect();}catch{}
    try{if(source)source.disconnect();}catch{}
    node=null;source=null;
    if(stream){
      for(const track of stream.getTracks())track.stop();
      stream=null;
    }
    if(context){const previous=context;context=null;previous.close().catch(()=>{});}
    $("voiceInput").disabled=false;
    $("voiceOutput").disabled=false;
    $("useWebSearch").disabled=false;
    button.setAttribute("aria-pressed","false");
    setVoiceButtonLabel("Lokaler Sprachmodus");
    resetVAD();
    say("Lokaler Sprachmodus aus · Audio bleibt auf diesem PC");
  }
  function fail(message){
    turnOff();say(message);
  }
  async function requestJSON(url,options={}){
    const response=await fetch(url,{credentials:"same-origin",cache:"no-store",...options});
    let packet;
    try{packet=await response.json()}catch{throw Error("Ungültige Antwort des lokalen Sprachservers.");}
    if(!response.ok)throw Error(packet.error||"Lokaler Sprachdienst nicht verfügbar.");
    return packet;
  }
  function waveFromBlocks(parts,rate){
    let length=0;
    for(const part of parts)length+=part.length;
    if(!length||!Number.isFinite(rate)||rate<8000||rate>192000)throw Error("Ungültige Mikrofonaufnahme.");
    const all=new Float32Array(length);
    let offset=0;
    for(const part of parts){all.set(part,offset);offset+=part.length;}
    const outLength=Math.min(MAX_SPEECH*SAMPLE_RATE,Math.floor(length*SAMPLE_RATE/rate));
    if(outLength<4000||outLength>MAX_SPEECH*SAMPLE_RATE)throw Error("Audio außerhalb des Zeitlimits.");
    const wav=new ArrayBuffer(44+outLength*2);
    const view=new DataView(wav);
    const label=(i,text)=>{
      for(let j=0;j<text.length;j++)view.setUint8(i+j,text.charCodeAt(j));
    };
    label(0,"RIFF");view.setUint32(4,36+outLength*2,true);
    label(8,"WAVE");label(12,"fmt ");
    view.setUint32(16,16,true);view.setUint16(20,1,true);
    view.setUint16(22,1,true);view.setUint32(24,SAMPLE_RATE,true);
    view.setUint32(28,SAMPLE_RATE*2,true);view.setUint16(32,2,true);
    view.setUint16(34,16,true);
    label(36,"data");view.setUint32(40,outLength*2,true);
    const ratio=rate/SAMPLE_RATE;
    for(let i=0;i<outLength;i++){
      const start=Math.floor(i*ratio);
      const end=Math.min(all.length,Math.max(start+1,Math.floor((i+1)*ratio)));
      let total=0;
      for(let j=start;j<end;j++)total+=all[j];
      const level=Math.max(-1,Math.min(1,total/(end-start)));
      view.setInt16(44+i*2,Math.round(level*(level<0?32768:32767)),true);
    }
    return new Blob([wav],{type:"audio/wav"});
  }
  async function finishUtterance(){
    if(!active||phase!=="listening"||!context)return;
    const current=version;
    phase="transcribing";
    for(const track of stream.getTracks())track.enabled=false;
    say("Sprache wird auf deinem PC erkannt …");
    const audioParts=blocks;
    resetVAD();
    controller=new AbortController();
    try {
      const blob=waveFromBlocks(audioParts,context.sampleRate);
      const r=await requestJSON("/api/neral/voice/transcribe",{
        method:"POST",
        headers:{"Content-Type":"audio/wav","X-Neral-CSRF":csrf},
        body:blob,signal:controller.signal
      });
      if(!active||current!==version)return;
      if(typeof r.text!=="string"||!r.local_only||!r.text.trim()){
        rearm("Keine Sprache erkannt · erneut sprechen");
        return;
      }
      const existing=prompt.value.trim();
      const updated=existing?existing+" "+r.text.trim():r.text.trim();
      if(new TextEncoder().encode(updated).length>4096){
        fail("Sprachtext überschreitet das Nachrichtenlimit.");
        return;
      }
      prompt.value=updated;
      if(existing){
        turnOff();
        say("Diktat ergänzt · Text prüfen und selbst absenden");
        return;
      }
      // sendMessage silently refuses a new request when a previous one is busy.
      // Never leave this voice session muted in an unobservable waiting state.
      if($("submit")?.disabled){
        turnOff();
        say("NERAL antwortet noch · Diktat im Eingabefeld prüfen und später absenden");
        return;
      }
      if(!form||typeof form.requestSubmit!=="function"){
        fail("Chatformular nicht verfügbar.");
        return;
      }
      awaiting=true;phase="waiting";
      say("NERAL antwortet lokal …");
      form.requestSubmit();
    } catch(e) {
      if(!active||current!==version)return;
      fail(e.message||"Lokale Spracherkennung fehlgeschlagen.");
    }
  }
  function consumePCM(data){
    if(!active||phase!=="listening"||!context)return;
    const block=data instanceof Float32Array?data:new Float32Array(data);
    if(!block.length)return;
    const duration=block.length/context.sampleRate;
    let sum=0;
    for(let i=0;i<block.length;i++)sum+=block[i]*block[i];
    const loud=Math.sqrt(sum/block.length)>.02;
    if(!heard){
      preRoll.push(block);
      if(preRoll.length>12)preRoll.shift();
      if(loud){heard=1;blocks.push(...preRoll);preRoll=[];seconds=blocks.reduce((n,v)=>n+v.length,0)/context.sampleRate;}
      return;
    }
    blocks.push(block);seconds+=duration;
    silence=loud?0:silence+duration;
    if(loud){heard++;say("Ich höre zu · zum Antworten kurz pausieren …");}
    if(silence>=END_SILENCE||seconds>=MAX_SPEECH)void finishUtterance();
  }
  function rearm(message){
    if(!active||!stream)return;
    resetVAD();phase="listening";awaiting=false;
    for(const track of stream.getTracks())track.enabled=true;
    say(message||"Lokales Mikrofon aktiv · sprich jetzt");
  }
  function segments(text){
    const result=[];let rest=String(text).slice(0,850);
    while(rest){
      let cut=Math.min(rest.length,240);
      if(rest.length>240){
        const gap=rest.slice(0,240).lastIndexOf(" ");
        if(gap>=130)cut=gap+1;
      }
      const chunk=rest.slice(0,cut).trim();
      if(chunk)result.push(chunk);
      rest=rest.slice(cut);
    }
    return result;
  }
  async function playAnswer(text){
    const current=version;
    phase="speaking";
    say("NERAL spricht · vollständig lokal");
    const chunks=segments(text);
    try {
      for(const chunk of chunks){
        if(!active||current!==version)return;
        controller=new AbortController();
        const r=await fetch("/api/neral/voice/speak",{
          credentials:"same-origin",cache:"no-store",method:"POST",
          headers:{"Content-Type":"application/json","X-Neral-CSRF":csrf},
          body:JSON.stringify({text:chunk}),signal:controller.signal
        });
        if(!r.ok){const error=await r.json().catch(()=>({}));
          throw Error(error.error||"Lokale deutsche Sprachsynthese fehlgeschlagen.");}
        const wav=await r.blob();
        if(!active||current!==version)return;
        const url=browser.URL.createObjectURL(wav);
        await new Promise((resolve,reject)=>{
          const audio=new browser.Audio(url);replyAudio=audio;
          let settled=false;
          const finish=error=>{
            if(settled)return;
            settled=true;
            audio.onended=null;
            audio.onerror=null;
            browser.URL.revokeObjectURL(url);
            if(replyAudio===audio)replyAudio=null;
            if(cancelPlayback===cancel)cancelPlayback=null;
            if(error)reject(error);
            else resolve();
          };
          const cancel=()=>{
            try{audio.pause();audio.currentTime=0;}catch{}
            finish();
          };
          cancelPlayback=cancel;
          audio.onended=()=>finish();
          audio.onerror=()=>finish(Error("Lokale WAV-Wiedergabe fehlgeschlagen."));
          try{Promise.resolve(audio.play()).catch(finish);}
          catch(error){finish(error);}
        });
      }
      if(active&&current===version)rearm("Du kannst weiterreden · lokales Mikrofon aktiv");
    }catch(e){if(active&&current===version)fail(e.message||"Lokales Vorlesen fehlgeschlagen.");}
  }
  async function start(){
    if(active||starting){turnOff();return;}
    if($("useWebSearch")?.checked){
      say("Websuche deaktivieren, bevor der vollständig lokale Sprachmodus startet.");
      return;
    }
    if(!(browser.AudioContext||browser.webkitAudioContext)){
      say("WebAudio ist in diesem Browser nicht verfügbar.");
      return;
    }
    const current=++version;
    starting=true;
    // Keep the same button clickable while the browser permission prompt
    // is open. A second click invalidates the pending getUserMedia result.
    setVoiceButtonLabel("Sprachstart abbrechen");
    say("Prüfe lokale Sprachmodelle …");
    try {
      const session=await requestJSON("/api/neral/preview/session");
      const state=await requestJSON("/api/neral/voice/status");
      if(current!==version)return;
      if(!state.ready||state.execution!=="local_cpu"||state.audio_leaves_pc!==false){
        say(state.reason || "Offline-Sprachmodelle fehlen · im NERAL-Ordner Install NERAL Voice.cmd starten, dann NERAL neu starten.");
        return;
      }
      if(!navigator.mediaDevices?.getUserMedia)
        throw Error("Mikrofonzugriff im Browser nicht verfügbar.");
      csrf=session.csrf;
      const media=await navigator.mediaDevices.getUserMedia({
        audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},
        video:false
      });
      if(current!==version){for(const track of media.getTracks())track.stop();return;}
      stream=media;
      const Context=browser.AudioContext||browser.webkitAudioContext;
      context=new Context();
      await context.resume();
      if(current!==version)return;
      source=context.createMediaStreamSource(stream);
      if(context.audioWorklet&&browser.AudioWorkletNode){
        await context.audioWorklet.addModule("/static/neral-local-voice-worklet.js");
        if(current!==version)return;
        node=new browser.AudioWorkletNode(context,"neral-local-capture");
        node.port.onmessage=e=>consumePCM(e.data);
      }else{
        // Older browsers still get local capture, never browser speech APIs.
        node=context.createScriptProcessor(2048,1,1);
        node.onaudioprocess=e=>{
          const data=e.inputBuffer.getChannelData(0);
          consumePCM(new Float32Array(data));
          for(let c=0;c<e.outputBuffer.numberOfChannels;c++)
            e.outputBuffer.getChannelData(c).fill(0);
        };
      }
      source.connect(node);node.connect(context.destination);
      active=true;
      $("voiceInput").disabled=true; // prevent browser-provider STT overlap
      $("voiceOutput").disabled=true;
      $("useWebSearch").disabled=true; // never send spoken text to a search provider
      button.setAttribute("aria-pressed","true");
      setVoiceButtonLabel("Lokalen Sprachmodus beenden");
      rearm();
    }catch(e){if(current===version)fail(e.message||"Lokales Mikrofon konnte nicht gestartet werden.");}
    finally{
      if(current===version){
        starting=false;
        if(!active)setVoiceButtonLabel("Lokaler Sprachmodus");
      }
      button.disabled=false;
    }
  }
  button.addEventListener("click",start);
  browser.addEventListener("neral:voice-answer",event=>{
    if(!active||!awaiting||phase!=="waiting")return;
    awaiting=false;
    const reply=event?.detail?.text;
    if(typeof reply==="string"&&reply.trim())void playAnswer(reply);
  });
  // The chat frontend reports failed/cancelled model turns through this
  // event. Recover from failed inference instead of waiting forever.
  browser.addEventListener("neral:activity",event=>{
    if(!active||!awaiting||phase!=="waiting")return;
    if(event?.detail?.phase==="error"||event?.detail?.phase==="cancelled")
      rearm("NERAL hat nicht geantwortet · erneut sprechen");
  });
  $("clear").addEventListener("click",turnOff);
  $("stopRequest").addEventListener("click",()=>{
    if(active||starting)turnOff();
  });
  $("stopVoice").addEventListener("click",()=>{
    if(active||starting)turnOff();
  });
  // Status is a local, read-only check: never request microphone permission
  // during page load, and never silently fetch any speech models.
  say("Prüfe lokale Sprachmodelle …");
  void requestJSON("/api/neral/voice/status").then(state=>{
    if(active||starting)return;
    if(state.ready && state.execution==="local_cpu" &&
       state.audio_leaves_pc===false){
      say("Lokaler Sprachmodus bereit · Mikrofon startet nur nach Klick");
    }else{
      say(state.reason || "Lokale Sprachmodelle fehlen · Install NERAL Voice.cmd im NERAL-Ordner ausführen, danach NERAL neu starten");
    }
  }).catch(()=>{
    if(!active&&!starting)
      say("Lokaler Sprachserver nicht erreichbar · NERAL auf diesem PC neu starten");
  });
})();
