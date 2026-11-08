"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const {TextEncoder}=require("node:util");

function harness() {
  const handlers=new Map(), calls=[], elements={};
  const element=id=>elements[id] ||= {
    value:"",checked:false,disabled:false,files:[],textContent:"",
    addEventListener(type,fn){handlers.set(id+":"+type,fn);},
    querySelector(){return null;},append(){},replaceChildren(){},remove(){},
    focus(){},setAttribute(){},scrollHeight:0,scrollTop:0,
    requestSubmit(){return handlers.get("chatForm:submit")({preventDefault(){}})}
  };
  element("chatForm").querySelector=()=>({value:"fast"});
  const fetch=async(url,opts={})=>{
    calls.push({url,opts});
    if(url.endsWith("/session")) return {ok:true,json:async()=>({csrf:"abc",model:"qwen3:8b"})};
    if(url.endsWith("/status")) return {ok:true,json:async()=>({model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false})};
    if(url==="/api/neral/chat/generate") return {ok:true,json:async()=>({text:"OK",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false})};
    throw Error("Unexpected URL "+url);
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8"),{
    document:{getElementById:element,createElement:()=>({className:"",textContent:"",append(){}})},
    window:{},fetch,TextEncoder,AbortSignal,Error,console
  },{timeout:2000});
  return {element,handlers,calls};
}
const flush=()=>new Promise(r=>setTimeout(r,10));

test("file context is loaded only through explicit file input and sent only with chat",async()=>{
  const x=harness();await flush();
  x.element("projectFiles").files=[{name:"math.py",size:33,text:async()=>"def answer():\n    return 42"}];
  await x.handlers.get("projectFiles:change")();
  assert.equal(x.calls.filter(y=>y.url==="/api/neral/chat/generate").length,0);
  assert.match(x.element("fileStatus").textContent,/math.py/);
  x.element("prompt").value="Prüfe die Funktion";
  await x.handlers.get("chatForm:submit")({preventDefault(){}});
  const sent=x.calls.find(y=>y.url==="/api/neral/chat/generate");
  assert.ok(sent);
  const data=JSON.parse(sent.opts.body);
  assert.match(data.prompt,/def answer/);
  assert.match(data.prompt,/math.py/);
  assert.ok(new TextEncoder().encode(data.prompt).length<=4096);
  assert.match(x.element("fileStatus").textContent,/keine|entfernt/i);
});

test("unsupported or oversized files are rejected without silently sending content",async()=>{
  const x=harness();await flush();
  x.element("projectFiles").files=[{name:"malware.exe",size:20000,text:async()=>{throw Error("must not read");}}];
  await x.handlers.get("projectFiles:change")();
  assert.match(x.element("fileStatus").textContent,/nicht|abgelehnt|groß/i);
  x.element("prompt").value="Guten Tag";
  await x.handlers.get("chatForm:submit")({preventDefault(){}});
  const sent=x.calls.find(y=>y.url==="/api/neral/chat/generate");
  assert.equal(JSON.parse(sent.opts.body).prompt,"Guten Tag");
});
