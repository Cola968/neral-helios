"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const vm=require("node:vm");
const fs=require("node:fs");
const path=require("node:path");
const {TextEncoder}=require("node:util");

test("a long model answer never poisons the following conversation history",async()=>{
  const handlers=new Map();
  const calls=[];
  const elements={};
  function element(id){
    return elements[id] ||= {
      value:"",textContent:"",disabled:false,
      addEventListener(kind,callback){handlers.set(id+":"+kind,callback);},
      querySelector(selector){if(id==="chatForm")return {value:"fast"};return null;},
      append(){},replaceChildren(){},remove(){},focus(){},setAttribute(){},
      scrollHeight:0,scrollTop:0
    };
  }
  element("prompt").value="erste Frage";
  let messagesSeen=0;
  const fetch=async(url,opts={})=>{
    if(url.endsWith("/session"))return {ok:true,json:async()=>({csrf:"LOCAL_CSRF",
                                                model:"qwen3:8b"})};
    if(url.endsWith("/status"))return {ok:true,json:async()=>({
       model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",
       adapter_loaded:false})};
    assert.equal(url,"/api/neral/chat/generate");
    calls.push(JSON.parse(opts.body));
    messagesSeen++;
    return {ok:true,json:async()=>({
       text: messagesSeen===1?"L".repeat(24000):"Berlin",
       model_origin:"local_foundation_not_neral_adapter",
       adapter_loaded:false,model_generated:true
    })};
  };
  const src=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  vm.runInNewContext(src,{
    document:{getElementById:element,createElement:tag=>({tag,textContent:"",append(){},remove(){},className:""})},
    fetch,TextEncoder,AbortSignal,Error,console
  },{timeout:2000});
  await new Promise(resolve=>setTimeout(resolve,5));
  const submit=handlers.get("chatForm:submit");
  assert.equal(typeof submit,"function");
  await submit({preventDefault(){}});
  element("prompt").value="zweite Frage";
  await submit({preventDefault(){}});
  assert.equal(calls.length,2);
  assert.deepEqual([...calls[1].history].map(x=>x.role),["user","assistant"]);
  assert.ok(new TextEncoder().encode(calls[1].history[1].content).length<=1900,
    "Large original model response must be reduced to <=1900 UTF-8 bytes");
  assert.ok(new TextEncoder().encode(JSON.stringify(calls[1].history)).length<=9000);
});
