"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");
const {TextEncoder} = require("node:util");

function browserHarness() {
  const handlers = new Map(), calls = [], speech = [];
  const elements = Object.create(null);
  const element = id => elements[id] ||= {
    value:"", checked:false, disabled:false, textContent:"", title:"",
    addEventListener(kind,cb){handlers.set(id+":"+kind,cb);},
    querySelector(selector){if(id==="chatForm") return {value:"fast"};return null;},
    append(){},replaceChildren(){},remove(){},focus(){},setAttribute(){},
    scrollHeight:0,scrollTop:0,requestSubmit(){
      const fn = handlers.get("chatForm:submit");
      if(fn) return fn({preventDefault(){}});
    }
  };
  element("voiceOutput").checked = true;
  element("voiceAutoSend").checked = true;
  element("prompt").value = "";
  class Recognition {
    constructor(){this.lang="";Recognition.last=this;}
    start(){this.started=true;}
    stop(){this.stopped=true;}
    abort(){this.aborted=true;}
  }
  const fetch = async (url,opts={})=>{
    calls.push({url,opts});
    if(url.endsWith("/session")) return {ok:true,json:async()=>({csrf:"CSRF",model:"qwen3:8b"})};
    if(url.endsWith("/status")) return {ok:true,json:async()=>({model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false})};
    if(url==="/api/web/search/status") return {ok:true,json:async()=>({configured:true,live_verified:false})};
    if(url==="/api/web/search") return {ok:true,json:async()=>({
      results:[{title:"Python documentation",snippet:"Python guide",url:"https://docs.python.org/3/"}]
    })};
    if(url==="/api/neral/chat/generate") return {ok:true,json:async()=>({
      text:"Antwort aus dem lokalen Modell", model_origin:"local_foundation_not_neral_adapter",
      adapter_loaded:false,model_generated:true
    })};
    throw new Error("Unexpected URL "+url);
  };
  const src = fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  const speak = {speak(utterance){speech.push(utterance.text);},cancel(){speech.push("CANCEL");}};
  vm.runInNewContext(src,{
    document:{getElementById:element,createElement:()=>({
      textContent:"",className:"",href:"",rel:"",target:"",
      append(){},remove(){},setAttribute(){}
    })},
    window:{SpeechRecognition:Recognition,speechSynthesis:speak},
    SpeechSynthesisUtterance:class {constructor(text){this.text=text;this.lang="";}},
    fetch,TextEncoder,AbortSignal,AbortController,Error,console
  },{timeout:2000});
  return {element,handlers,calls,speech,Recognition};
}
const flush=()=>new Promise(resolve=>setTimeout(resolve,10));

test("microphone requires a click; opt-in transcript reaches the local chat and TTS",async()=>{
  const ctx=browserHarness();
  await flush();
  assert.equal(ctx.Recognition.last,undefined, "Do not open microphone automatically");
  const start=ctx.handlers.get("voiceInput:click");
  assert.equal(typeof start,"function");
  start();
  assert.equal(ctx.Recognition.last.started,true);
  ctx.Recognition.last.onresult({results:[[ {transcript:"Hallo NERAL"} ]]});
  await flush();
  const sent=ctx.calls.find(x=>x.url==="/api/neral/chat/generate");
  assert.ok(sent,"Speech transcript must reach local chat");
  assert.equal(JSON.parse(sent.opts.body).prompt,"Hallo NERAL");
  assert.ok(ctx.speech.includes("Antwort aus dem lokalen Modell"));
});

test("web plugin stays off unless selected and searches only after submission",async()=>{
  const ctx=browserHarness();
  await flush();
  assert.equal(ctx.calls.filter(x=>x.url==="/api/web/search").length,0);
  ctx.element("useWebSearch").checked=true;
  ctx.element("prompt").value="Suche Python documentation";
  await ctx.handlers.get("chatForm:submit")({preventDefault(){}});
  assert.equal(ctx.calls.filter(x=>x.url==="/api/web/search").length,1);
  const sent=ctx.calls.find(x=>x.url==="/api/neral/chat/generate");
  assert.ok(sent);
  const msg=JSON.parse(sent.opts.body);
  assert.ok(msg.prompt.includes("https://docs.python.org/3/"));
  assert.ok(msg.prompt.includes("Suche Python documentation"));
  assert.ok(!msg.history.length);
});

test("stop voice cancels speech and a missing browser speech API disables mic",async()=>{
  const ctx=browserHarness();
  await flush();
  ctx.handlers.get("stopVoice:click")();
  assert.ok(ctx.speech.includes("CANCEL"));
});

test("opt-in local Piper speech suppresses simultaneous browser speech synthesis",async()=>{
  const ctx=browserHarness();
  await flush();
  // Voice v2 keeps the legacy browser readout enabled for normal chat.
  ctx.element("localVoiceStart").getAttribute=key=>
    key==="aria-pressed"?"true":null;
  ctx.element("prompt").value="Hallo NERAL";
  await ctx.handlers.get("chatForm:submit")({preventDefault(){}});
  assert.ok(ctx.calls.find(x=>x.url==="/api/neral/chat/generate"));
  assert.equal(ctx.speech.filter(x=>x!=="CANCEL").length,0,
    "Only Piper may speak while local voice is active");
});
