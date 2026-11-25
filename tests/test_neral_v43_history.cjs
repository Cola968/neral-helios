"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const {TextEncoder}=require("node:util");
function harness(){
  const handlers=new Map(),events=new Map(),elements={},calls=[];
  function el(id){return elements[id] ||= {value:"",checked:false,disabled:false,files:[],textContent:"",
    children:[],scrollHeight:0,scrollTop:0,
    addEventListener(n,f){handlers.set(id+":"+n,f)},querySelector(){return id==="chatForm"?{value:"fast"}:null},
    append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x},remove(){},focus(){},setAttribute(){}};
  }
  const browser={addEventListener(n,f){events.set(n,[...(events.get(n)||[]),f])},dispatchEvent(e){for(const f of events.get(e.type)||[])f(e)}};
  class CustomEvent{constructor(type,o={}){this.type=type;this.detail=o.detail}}
  const fetch=async(url,opts={})=>{
    if(url.endsWith("/session"))return {ok:true,json:async()=>({csrf:"local",model:"qwen3:8b"})};
    if(url.endsWith("/status"))return {ok:true,json:async()=>({model:"qwen3:8b",adapter_loaded:false,model_origin:"local_foundation_not_neral_adapter"})};
    calls.push(JSON.parse(opts.body));
    return {ok:true,json:async()=>({text:"Antwort",model_origin:"local_foundation_not_neral_adapter",model_generated:true,adapter_loaded:false})};
  };
  const source=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  vm.runInNewContext(source,{window:browser,CustomEvent,document:{getElementById:el,createElement:tag=>({tag,className:"",textContent:"",children:[],append(...x){this.children.push(...x)},remove(){}})},fetch,TextEncoder,AbortSignal,AbortController,Error,console},{timeout:2000});
  return {el,calls,handlers,emit:(n,detail)=>browser.dispatchEvent(new CustomEvent(n,{detail})),flush:()=>new Promise(r=>setImmediate(r))};
}
test("incomplete restored turns never poison the next chat history",async()=>{
  const h=harness();await h.flush();
  h.emit("neral:load-chat",{messages:[
    {role:"assistant",content:"Orphan"},
    {role:"user",content:"Unfinished attempt"},
    {role:"user",content:"Completed question"},
    {role:"assistant",content:"Completed answer"},
    {role:"assistant",content:"Orphan 2"},
    {role:"user",content:"Unfinished tail"}
  ]});
  h.el("prompt").value="Next question";
  await h.handlers.get("chatForm:submit")({preventDefault(){}});
  assert.equal(h.calls.length,1);
  assert.deepEqual(h.calls[0].history,[{role:"user",content:"Completed question"},{role:"assistant",content:"Completed answer"}]);
});
test("manual web and forced sandbox are mutually exclusive",async()=>{
  const h=harness();await h.flush();
  h.el("forceSandbox").checked=true;
  h.el("useWebSearch").checked=true;
  h.handlers.get("useWebSearch:change")();
  assert.equal(h.el("forceSandbox").checked,false);
  h.el("forceSandbox").checked=true;
  h.handlers.get("forceSandbox:change")();
  assert.equal(h.el("useWebSearch").checked,false);
});
