"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const {TextEncoder}=require("node:util");

async function runPrompt(question,{web=false,results=[]}={}){
  const handlers=new Map(),elements={},calls=[];
  function element(id){
    return elements[id] ||= {value:"",textContent:"",checked:false,disabled:false,children:[],
      addEventListener(kind,fn){handlers.set(id+":"+kind,fn);},
      querySelector(){return id==="chatForm"?{value:"fast"}:null;},
      append(...x){this.children.push(...x);},replaceChildren(...x){this.children=x;},
      remove(){},focus(){},setAttribute(){},getAttribute(){return null;},
      scrollHeight:0,scrollTop:0};
  }
  element("prompt").value=question;
  element("useWebSearch").checked=web;
  async function fetch(url,options={}){
    calls.push({url,body:options.body&&typeof options.body==="string"?JSON.parse(options.body):null});
    if(url.endsWith("/session"))
      return {ok:true,json:async()=>({csrf:"TEST_CSRF",model:"qwen3:8b"})};
    if(url.endsWith("/status"))
      return {ok:true,json:async()=>({model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false})};
    if(url==="/api/web/search")
      return {ok:true,json:async()=>({results,provider:"fixture"})};
    if(url==="/api/neral/chat/generate")
      return {ok:true,json:async()=>({text:"Antwort",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false,model_generated:true})};
    throw new Error("Unexpected URL "+url);
  }
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8"),{
    document:{getElementById:element,createElement:tag=>({tag,textContent:"",children:[],className:"",
      append(...x){this.children.push(...x)},remove(){},setAttribute(){},getAttribute(){return null}})},
    fetch,TextEncoder,AbortSignal,AbortController,Error,console,
  },{timeout:2000});
  await new Promise(resolve=>setTimeout(resolve,10));
  await handlers.get("chatForm:submit")({preventDefault(){}});
  return {calls,elements};
}

test("explicit internet request with Web disabled shows a useful message and never calls model",async()=>{
  const {calls,elements}=await runPrompt("Suche im Internet nach aktuellen Raumfahrtmissionen",{web:false});
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,0);
  assert.equal(calls.filter(c=>c.url==="/api/neral/chat/generate").length,0);
  assert.match(elements.error.textContent,/Web.*aktivier|aktivier.*Web/i);
  assert.equal(elements.prompt.value,"Suche im Internet nach aktuellen Raumfahrtmissionen");
});

test("search with zero relevant sources informs model that a search actually ran",async()=>{
  const question="Suche im Internet nach aktuellen Raumfahrtmissionen";
  const {calls}=await runPrompt(question,{web:true,results:[]});
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,1);
  const modelCall=calls.find(c=>c.url==="/api/neral/chat/generate");
  assert.ok(modelCall,"the local model still receives the question");
  assert.match(modelCall.body.prompt,/NERAL-WEBSTATUS/);
  assert.match(modelCall.body.prompt,/keine passenden Treffer/);
  assert.match(modelCall.body.prompt,/URSPRÜNGLICHE FRAGE/);
});

test("plain imperative Suche nach ... requires Web consent, not a model refusal", async()=>{
  const question="suche nach dem berliner flughafen";
  const {calls,elements}=await runPrompt(question,{web:false});
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,0);
  assert.equal(calls.filter(c=>c.url==="/api/neral/chat/generate").length,0);
  assert.match(elements.error.textContent,/Web.*aktivier|aktivier.*Web/i);
  assert.equal(elements.prompt.value,question);
});
