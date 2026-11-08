"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const vm=require("node:vm");
const fs=require("node:fs");
const path=require("node:path");
const {TextEncoder}=require("node:util");

function setup({ignoreAbort=false}={}){
  const handlers=new Map(),nodes={};
  const messages=[];
  const events=[];
  const calls=[];
  let pending=[];
  function element(id){
    if(nodes[id])return nodes[id];
    const node={
      value:"",textContent:"",disabled:false,className:"",children:[],
      addEventListener(kind,callback){handlers.set(id+":"+kind,callback);},
      querySelector(selector){return id==="chatForm" ? {value:"fast"} : null;},
      append(...children){this.children.push(...children);if(id==="messages")messages.push(...children);},
      replaceChildren(){this.children.length=0;if(id==="messages")messages.length=0;},
      remove(){},focus(){},scrollHeight:0,scrollTop:0
    };
    nodes[id]=node;
    return node;
  }
  element("prompt").value="Please answer";
  const fetch=async(url,options={})=>{
    if(url.endsWith("/session"))return {ok:true,json:async()=>({csrf:"LOCAL_CSRF",model:"qwen3:8b"})};
    if(url.endsWith("/status"))return {ok:true,json:async()=>({
      model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",
      adapter_loaded:false
    })};
    calls.push({url,options,data:JSON.parse(options.body)});
    return new Promise((resolve,reject)=>{
      const req={resolve,reject,options};
      pending.push(req);
      if(!ignoreAbort){
        options.signal.addEventListener("abort",()=>reject(Object.assign(new Error("Aborted"),{name:"AbortError"})),{once:true});
      }
    });
  };
  const source=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  const context={document:{getElementById:element,
                  createElement:tag=>({tagName:tag.toUpperCase(),textContent:"",children:[],
                                       append(...x){this.children.push(...x);}})},
                 fetch,TextEncoder,AbortSignal,AbortController,console,Error};
  vm.runInNewContext(source,context,{timeout:2000});
  const submit=handlers.get("chatForm:submit");
  const stop=handlers.get("stop:click");
  const clear=handlers.get("clear:click");
  return {handlers,nodes,messages,events,calls,pending,
          submit,stop,clear,settle:()=>new Promise(resolve=>setImmediate(resolve))};
}

test("an explicit Stop control aborts HTTP request without falsely claiming model GPU stopped",async()=>{
  const html=fs.readFileSync(path.join(__dirname,"../templates/neral_preview.html"),"utf8");
  assert.match(html,/id="stop"/);
  const u=setup();
  await u.settle();
  assert.equal(typeof u.stop,"function");
  assert.equal(u.nodes.stop.disabled,true);
  const request=u.submit({preventDefault(){}});
  assert.equal(u.nodes.stop.disabled,false);
  assert.equal(u.calls.length,1);
  u.stop();
  await request;
  assert.equal(u.calls[0].options.signal.aborted,true);
  assert.equal(u.nodes.stop.disabled,true);
  assert.equal(u.nodes.submit.disabled,false);
  assert.equal(u.messages.length,0,"Aborted model reply must not be inserted");
  assert.match(u.nodes.error.textContent,/Ollama|Berechnung|weiterlaufen/i);
  assert.equal(u.nodes.prompt.value,"Please answer","Failed requests preserve user input");
});

test("clearing an in-flight chat must discard a late response even if fetch ignores cancellation",async()=>{
  const u=setup({ignoreAbort:true});
  await u.settle();
  const request=u.submit({preventDefault(){}});
  assert.equal(u.calls.length,1);
  u.clear();
  u.pending[0].resolve({ok:true,json:async()=>({
    text:"OLD_GENERATION_SHOULD_NOT_APPEAR",
    model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false,
    model_generated:true
  })});
  await request;
  assert.equal(u.calls[0].options.signal.aborted,true);
  assert.equal(u.messages.some(item=>JSON.stringify(item).includes("OLD_GENERATION_SHOULD_NOT_APPEAR")),false);
  assert.equal(u.nodes.prompt.value,"Please answer");
  u.nodes.prompt.value="Neue Frage";
  const next=u.submit({preventDefault(){}});
  assert.equal(u.calls.length,2);
  assert.equal(u.calls[1].data.history.length,0,"Cleared old chat must not remain in model history");
  u.pending[1].resolve({ok:true,json:async()=>({
    text:"NEUE ANTWORT",model_origin:"local_foundation_not_neral_adapter",
    adapter_loaded:false,model_generated:true
  })});
  await next;
  assert.equal(u.messages.some(item=>JSON.stringify(item).includes("NEUE ANTWORT")),true);
});
