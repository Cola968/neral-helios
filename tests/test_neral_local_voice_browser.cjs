"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const vm=require("node:vm");
const fs=require("node:fs");
const path=require("node:path");

function setup({ready=true,transcript="Hallo NERAL",reason="",microphoneGate=null}={}){
  const events=new Map(), elems=new Map(), requests=[], tracks=[], worklets=[], audios=[], objectURLs=[];
  const element=id=>{
    if(!elems.has(id))elems.set(id,{
      textContent:"",value:"",disabled:false,checked:true,
      addEventListener(kind,cb){events.set(id+":"+kind,cb)},
      requestSubmit(){this.submitted=(this.submitted||0)+1},
      focus(){},setAttribute(){},getAttribute(){return "false"},
    });
    return elems.get(id);
  };
  element("useWebSearch").checked=false;
  const track={enabled:true, stop(){this.stopped=true}};
  const stream={getTracks(){return [track]}};
  let micCount=0;
  class AudioContext{
    constructor(){this.sampleRate=48000;this.destination={};this.audioWorklet={addModule:async(url)=>{worklets.push(url)}};AudioContext.latest=this;}
    createMediaStreamSource(){return {connect(){},disconnect(){}}}
    close(){this.closed=true;return Promise.resolve()}
    resume(){return Promise.resolve()}
  }
  class AudioWorkletNode{
    constructor(){this.port={onmessage:null};AudioWorkletNode.latest=this;}
    connect(){}
    disconnect(){}
  }
  class Audio{
    constructor(url){this.url=url;audios.push(this)}
    play(){return Promise.resolve()}
    pause(){this.paused=true}
  }
  const win={AudioContext,AudioWorkletNode,Audio,URL:{
    createObjectURL(){
      const url="blob:audio-"+objectURLs.length;
      objectURLs.push({url,revoked:false});
      return url;
    },
    revokeObjectURL(url){
      const entry=objectURLs.find(item=>item.url===url);
      if(entry)entry.revoked=true;
    },
  },addEventListener(name,cb){events.set("window:"+name,cb)}};
  const navigator={mediaDevices:{async getUserMedia(){micCount++;if(microphoneGate)await microphoneGate();tracks.push(track);return stream}}};
  const fetch=async(url,options={})=>{
    requests.push({url,options});
    if(url.endsWith("/status"))return {ok:true,json:async()=>({
      stt_ready:ready,tts_ready:ready,ready,execution:"local_cpu",audio_leaves_pc:false,reason})};
    if(url.endsWith("/session"))return {ok:true,json:async()=>({csrf:"x".repeat(32)})};
    if(url.endsWith("/transcribe"))return {ok:true,json:async()=>({text:transcript,local_only:true})};
    if(url.endsWith("/speak"))return {ok:true,blob:async()=>({size:420})};
    throw Error("unexpected "+url);
  };
  const src=fs.readFileSync(path.join(__dirname,"../static/neral-local-voice.js"),"utf8");
  vm.runInNewContext(src,{document:{getElementById:element},window:win,navigator,
                       fetch,Audio,URL:win.URL,AbortController,Float32Array,
                       ArrayBuffer,DataView,Blob,Math,Number,TextEncoder,console},
                     {timeout:2000});
  return {element,events,requests,tracks,worklets,audios,objectURLs,win,
          micCount:()=>micCount,node:()=>AudioWorkletNode.latest,
          context:()=>AudioContext.latest};
}
const flush=async()=>{await new Promise(resolve=>setTimeout(resolve,20))};
const pulse=async(set,level,seconds)=>{
  const node=set.node(),count=Math.ceil(seconds*48000/2048);
  for(let i=0;i<count;i++){
    node.port.onmessage({data:new Float32Array(2048).fill(level)});
  }
  await flush();
};

test("offline voice never opens mic before explicit click, never calls Web Speech",async()=>{
  const s=setup();await flush();
  assert.equal(s.micCount(),0);
  assert.equal(s.requests.filter(x=>x.url.includes("transcribe")).length,0);
  await s.events.get("localVoiceStart:click")();
  assert.equal(s.micCount(),1);
  assert.ok(s.worklets.includes("/static/neral-local-voice-worklet.js"));
  assert.equal(s.element("voiceInput").disabled,true,"prevent mixing browser provider");
  assert.match(s.element("localVoiceStatus").textContent,/lokal|sprich/i);
});

test("missing offline models fail closed without microphone or cloud fallback",async()=>{
  const s=setup({ready:false});await flush();
  await s.events.get("localVoiceStart:click")();
  assert.equal(s.micCount(),0);
  assert.match(s.element("localVoiceStatus").textContent,/install|fehl/i);
});

test("silence finalizes one turn, posts PCM to localhost with CSRF, uses chat UI",async()=>{
  const s=setup();await flush();
  await s.events.get("localVoiceStart:click")();
  await pulse(s,.1,.5);
  await pulse(s,0,.9);
  const req=s.requests.filter(x=>x.url.endsWith("/transcribe"));
  assert.equal(req.length,1);
  assert.equal(req[0].options.headers["Content-Type"],"audio/wav");
  assert.equal(req[0].options.headers["X-Neral-CSRF"],"x".repeat(32));
  assert.equal(s.element("prompt").value,"Hallo NERAL");
  assert.equal(s.element("chatForm").submitted,1);
  assert.equal(s.tracks[0].enabled,false);
  const raw=new Uint8Array(await req[0].options.body.arrayBuffer());
  assert.equal(String.fromCharCode(...raw.slice(0,4)),"RIFF");
});

test("local assistant reply uses Piper WAV, never browser speech synthesis",async()=>{
  const s=setup();await flush();
  await s.events.get("localVoiceStart:click")();
  await pulse(s,.1,.5);await pulse(s,0,.9);
  s.events.get("window:neral:voice-answer")({detail:{text:"Guten Tag."}});
  await flush();
  assert.equal(s.requests.filter(x=>x.url.endsWith("/speak")).length,1);
  assert.equal(s.audios.length,1);
  s.audios[0].onended();
  await flush();
  assert.equal(s.tracks[0].enabled,true,"after local TTS, rearm for next utterance");
});

test("stop and clear discard delayed STT/TTS and release real mic tracks",async()=>{
  const s=setup();await flush();
  await s.events.get("localVoiceStart:click")();
  const old=s.node().port.onmessage;
  await s.events.get("localVoiceStart:click")();
  assert.equal(s.tracks[0].stopped,true);
  old({data:new Float32Array(2048).fill(.1)});
  await flush();
  assert.equal(s.requests.filter(x=>x.url.endsWith("/transcribe")).length,0);
  assert.equal(s.element("voiceInput").disabled,false);
  assert.equal(s.element("voiceOutput").disabled,false);
});

test("existing typed draft is never silently auto-submitted by voice",async()=>{
  const s=setup();await flush();
  s.element("prompt").value="Bitte prüfen:";
  await s.events.get("localVoiceStart:click")();
  await pulse(s,.1,.5);await pulse(s,0,.9);
  assert.equal(s.element("prompt").value,"Bitte prüfen: Hallo NERAL");
  assert.equal(s.element("chatForm").submitted||0,0);
  assert.equal(s.tracks[0].stopped,true,"exit auto voice when user must review text");
});

test("offline voice refuses external web search before microphone consent",async()=>{
  const s=setup();await flush();
  s.element("useWebSearch").checked=true;
  await s.events.get("localVoiceStart:click")();
  assert.equal(s.micCount(),0,"Search-enabled speech must not upload an implicit query");
  assert.match(s.element("localVoiceStatus").textContent,/Websuche.*deaktivier/i);
});

test("cancelling a running NERAL chat stops the offline voice session",async()=>{
  const s=setup();await flush();
  await s.events.get("localVoiceStart:click")();
  await pulse(s,.1,.5);await pulse(s,0,.9);
  assert.equal(s.element("chatForm").submitted,1);
  s.events.get("stopRequest:click")();
  assert.equal(s.tracks[0].stopped,true);
  s.events.get("window:neral:voice-answer")({detail:{text:"Späte Antwort"}});
  await flush();
  assert.equal(s.requests.filter(x=>x.url.endsWith("/speak")).length,0);
});

test("offline voice explains its actual install step on load without touching the mic",async()=>{
  const missing=setup({ready:false});
  await flush();
  assert.equal(missing.micCount(),0);
  assert.match(missing.element("localVoiceStatus").textContent,/Install NERAL Voice\.cmd/);
  const prepared=setup({ready:true});
  await flush();
  assert.equal(prepared.micCount(),0);
  assert.match(prepared.element("localVoiceStatus").textContent,/bereit|verfuegbar|verfügbar/i);
});

test("unsupported frozen EXE shows safe ZIP-only explanation without accessing mic",async()=>{
  const reason="NERAL.exe hat keine Offline-Sprachpakete. Bitte die ZIP-Version verwenden.";
  const s=setup({ready:false,reason});
  await flush();
  assert.equal(s.element("localVoiceStatus").textContent,reason);
  assert.equal(s.micCount(),0);
  await s.events.get("localVoiceStart:click")();
  assert.equal(s.micCount(),0);
  assert.equal(s.element("localVoiceStatus").textContent,reason);
});

test("failed model response reopens local microphone instead of hanging in waiting",async()=>{
  const s=setup();await flush();
  await s.events.get("localVoiceStart:click")();
  await pulse(s,.1,.5);await pulse(s,0,.9);
  assert.equal(s.tracks[0].enabled,false);
  s.events.get("window:neral:activity")({detail:{phase:"error"}});
  assert.equal(s.tracks[0].enabled,true);
  assert.match(s.element("localVoiceStatus").textContent,/erneut|weiter|sprich/i);
});

test("stopping Piper playback releases object URL and stops audio",async()=>{
  const s=setup();await flush();
  await s.events.get("localVoiceStart:click")();
  await pulse(s,.1,.5);await pulse(s,0,.9);
  s.events.get("window:neral:voice-answer")({detail:{text:"Guten Tag."}});
  await flush();
  assert.equal(s.objectURLs.length,1);
  const audio=s.audios[0];
  s.events.get("localVoiceStart:click")();
  assert.equal(audio.paused,true);
  assert.equal(s.objectURLs[0].revoked,true,"object URL must be released on explicit stop");
  audio.onended?.();
  await flush();
  assert.equal(s.tracks[0].stopped,true);
});

test("a busy chat preserves recognized dictation without auto-submitting or hanging",async()=>{
  const s=setup();await flush();
  await s.events.get("localVoiceStart:click")();
  s.element("submit").disabled=true;
  await pulse(s,.1,.5);await pulse(s,0,.9);
  assert.equal(s.element("prompt").value,"Hallo NERAL");
  assert.equal(s.element("chatForm").submitted||0,0);
  assert.equal(s.tracks[0].stopped,true,"dictation must be available for manual sending");
});

test("startup can be canceled while permission prompt is pending",async()=>{
  let grant;
  const gate=new Promise(resolve=>{grant=resolve});
  const s=setup({microphoneGate:()=>gate});await flush();
  const pending=s.events.get("localVoiceStart:click")();
  await flush();
  assert.equal(s.micCount(),1);
  assert.equal(s.element("localVoiceStart").disabled,false,"cancel control must remain clickable");
  await s.events.get("localVoiceStart:click")();
  grant();
  await pending;
  assert.equal(s.tracks[0].stopped,true,"late permission must not reopen the mic");
  assert.equal(s.element("localVoiceStart")["aria-pressed"],"false");
});
