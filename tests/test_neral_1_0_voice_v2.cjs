"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const vm=require("node:vm");
const fs=require("node:fs");
const path=require("node:path");
const {TextEncoder}=require("node:util");

function launch({answer="Antwort von NERAL",recognition=true}={}){
  const handlers=new Map(),calls=[],utterances=[],elements={};
  const element=id=>elements[id]||=( {
    value:"",textContent:"",checked:false,disabled:false,title:"",
    addEventListener(type,cb){handlers.set(id+":"+type,cb);},
    setAttribute(k,v){this[k]=v;},getAttribute(k){return this[k];},
    querySelector(){return id==="chatForm"?{value:"fast"}:null;},
    append(){},replaceChildren(){},remove(){},focus(){},
    requestSubmit(){const cb=handlers.get("chatForm:submit");return cb&&cb({preventDefault(){}});},
    scrollTop:0,scrollHeight:0
  });
  element("voiceAutoSend").checked=true;
  element("voiceOutput").checked=true;
  element("voiceSpeed").value="1";
  const recognizers=[];
  class Recognition {
    constructor(){recognizers.push(this);}
    start(){this.started=true;}
    abort(){this.aborted=true;}
    stop(){this.stopped=true;}
  }
  const synthesis={
    cancelCount:0, cancel(){this.cancelCount++;},
    speak(u){utterances.push(u);},
    getVoices(){return [{name:"Deutsch Lokal",lang:"de-DE",localService:true}]},
  };
  const fetch=async(url,init={})=>{
    calls.push({url,init});
    if(url.endsWith("/session"))return {ok:true,json:async()=>({csrf:"TOKEN",model:"qwen3:8b"})};
    if(url.endsWith("/status"))return {ok:true,json:async()=>({
      model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",
      adapter_loaded:false})};
    if(url==="/api/neral/chat/generate")return {ok:true,json:async()=>({
      text:answer,model_origin:"local_foundation_not_neral_adapter",
      adapter_loaded:false,model_generated:true})};
    throw Error("Unexpected URL "+url);
  };
  const browser={speechSynthesis:synthesis};
  if(recognition) browser.SpeechRecognition=Recognition;
  const src=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  vm.runInNewContext(src,{
    document:{getElementById:element,createElement:()=>({
      value:"",textContent:"",append(){},remove(){},setAttribute(){},
      getAttribute(){return "false";},addEventListener(){}})},
    window:browser,navigator:{clipboard:{writeText:async()=>{}}},
    SpeechSynthesisUtterance:class {constructor(text){this.text=text;}},
    fetch,TextEncoder,AbortSignal,AbortController,Error,console,
  },{timeout:2500});
  return {element,handlers,recognizers,calls,utterances,synthesis};
}
const flush=()=>new Promise(resolve=>setTimeout(resolve,20));
const chatCalls=ctx=>ctx.calls.filter(v=>v.url==="/api/neral/chat/generate");
const final=t=>Object.assign([{transcript:t}],{isFinal:true});
const interim=t=>Object.assign([{transcript:t}],{isFinal:false});

test("live interim transcript visible, never sent; only final speech submits once",async()=>{
  const c=launch();await flush();
  assert.equal(c.recognizers.length,0,"No surprise microphone opening");
  c.handlers.get("voiceInput:click")();
  const r=c.recognizers[0];
  assert.equal(r.interimResults,true,"Live recognition must be requested");
  r.onresult({results:[interim("Hallo Ner")]});
  assert.match(c.element("voiceStatus").textContent,/Hallo Ner/);
  assert.equal(c.element("prompt").value,"");
  assert.equal(chatCalls(c).length,0,"Interim cannot auto-send");
  r.onresult({results:[final("Hallo NERAL")]});
  await flush();
  assert.equal(chatCalls(c).length,1);
  assert.equal(JSON.parse(chatCalls(c)[0].init.body).prompt,"Hallo NERAL");
  r.onresult({results:[final("Hallo NERAL")]});
  await flush();
  assert.equal(chatCalls(c).length,1,"Duplicated final speech must not submit twice");
});

test("stop and clear invalidate delayed results even after browser callbacks",async()=>{
  const c=launch();await flush();
  c.handlers.get("voiceInput:click")();
  const old=c.recognizers[0];
  c.handlers.get("voiceInput:click")();
  assert.equal(old.aborted,true,"Cancel immediately, don't flush unwanted final transcript");
  old.onresult({results:[final("nicht absenden")]});
  assert.equal(chatCalls(c).length,0);
  assert.equal(c.element("prompt").value,"");
  c.handlers.get("voiceInput:click")();
  const second=c.recognizers[1];
  c.handlers.get("clear:click")();
  second.onresult({results:[final("nach Verlauf löschen")]});
  await flush();
  assert.equal(chatCalls(c).length,0);
  assert.equal(c.element("prompt").value,"");
});

test("voice must preserve typed draft and require manual review before sending",async()=>{
  const c=launch();await flush();
  c.element("prompt").value="Bitte nur erklären:";
  c.handlers.get("voiceInput:click")();
  c.recognizers[0].onresult({results:[final("ohne Code")]});
  await flush();
  assert.equal(c.element("prompt").value,"Bitte nur erklären: ohne Code");
  assert.equal(chatCalls(c).length,0,"Do not auto-send user's preexisting draft");
  assert.match(c.element("voiceStatus").textContent,/prüfen|Senden/i);
});

test("voice readout speaks across segments, applies speed and cannot resume after stop",async()=>{
  const longAnswer=Array.from({length:520},(_,i)=>"Wort"+i+".").join(" ");
  assert.ok(longAnswer.length>1800);
  const c=launch({answer:longAnswer});await flush();
  c.element("voiceSpeed").value="0.85";
  c.element("prompt").value="Bitte erzählen";
  await c.handlers.get("chatForm:submit")({preventDefault(){}});
  assert.equal(chatCalls(c).length,1);
  assert.equal(c.utterances.length,1);
  assert.ok(c.utterances[0].text.length<=500);
  assert.equal(c.utterances[0].rate,0.85);
  for(let i=0;i<30&&c.utterances.length===i+1;i++){
    const previous=c.utterances[i];
    assert.equal(typeof previous.onend,"function");
    previous.onend();
  }
  assert.ok(c.utterances.length>=4,"Long replies are no longer cut at 1800 chars");
  assert.ok(c.utterances.map(u=>u.text).join(" ").length>1800);
  const last=c.utterances.at(-1);
  const before=c.utterances.length;
  c.handlers.get("stopVoice:click")();
  last.onend?.();
  assert.equal(c.utterances.length,before,"A cancelled segment must not queue more audio");
});

test("speech errors are surfaced without console secrets or auto-open microphone",async()=>{
  const c=launch();await flush();
  c.handlers.get("voiceInput:click")();
  c.recognizers[0].onerror({error:"not-allowed"});
  assert.match(c.element("voiceStatus").textContent,/verweigert|Berechtigung|fehlgeschlagen/i);
  assert.equal(chatCalls(c).length,0);
  assert.equal(c.recognizers.length,1);
  const noMic=launch({recognition:false});
  assert.equal(noMic.element("voiceInput").disabled,true);
});
