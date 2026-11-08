"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const vm=require("node:vm");
const fs=require("node:fs");
const path=require("node:path");

function setup({ready=true,transcript="Hallo NERAL"}={}){
  const events=new Map(), elems=new Map(), requests=[], tracks=[], worklets=[], audios=[];
  const element=id=>{
    if(!elems.has(id))elems.set(id,{
      textContent:"",value:"",disabled:false,checked:true,
      addEventListener(kind,cb){events.set(id+":"+kind,cb)},
      requestSubmit(){this.submitted=(this.submitted||0)+1},
      focus(){},setAttribute(){},getAttribute(){return "false"},
    });
    return elems.get(id);
  };
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
    createObjectURL(){return "blob:audio"}, revokeObjectURL(){},
  },addEventListener(name,cb){events.set("window:"+name,cb)}};
  const navigator={mediaDevices:{async getUserMedia(){micCount++;tracks.push(track);return stream}}};
  const fetch=async(url,options={})=>{
    requests.push({url,options});
    if(url.endsWith("/status"))return {ok:true,json:async()=>({
      stt_ready:ready,tts_ready:ready,ready,execution:"local_cpu",audio_leaves_pc:false})};
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
  return {element,events,requests,tracks,worklets,audios,win,
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
