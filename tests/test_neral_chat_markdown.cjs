"use strict";
const test=require("node:test"),assert=require("node:assert/strict"),vm=require("node:vm");
const fs=require("node:fs"),path=require("node:path");
const {TextEncoder}=require("node:util");

test("assistant Markdown is readable and escaped HTML remains inert",async()=>{
  const handlers=new Map(),els={},created=[];
  function element(id){return els[id] ||= {
    value:"",checked:false,textContent:"",disabled:false,
    addEventListener(kind,fn){handlers.set(id+":"+kind,fn)},
    querySelector(){return {value:"fast"}},querySelectorAll(){return []},
    append(){},replaceChildren(){},focus(){},remove(){},setAttribute(){},
    scrollHeight:0,scrollTop:0
  }}
  element("prompt").value="Was ist die Hauptstadt von Schweden?";
  const fetch=async(url)=>({ok:true,json:async()=>{
    if(url.endsWith("/session"))return {csrf:"TEST",model:"qwen3:8b"};
    if(url.endsWith("/status"))return {model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false};
    return {text:"**Stockholm** &#x20; <img src=x onerror=alert(1)>",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false,model_generated:true};
  }});
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8"),{
    document:{getElementById:element,createElement:tag=>{
      const node={tag,textContent:"",children:[],append(...xs){this.children.push(...xs)},remove(){},setAttribute(){}};
      created.push(node);return node;
    }},fetch,TextEncoder,AbortSignal,AbortController,Error,console
  },{timeout:2000});
  await new Promise(resolve=>setTimeout(resolve,5));
  await handlers.get("chatForm:submit")({preventDefault(){}});
  const bold=created.find(n=>n.tag==="b");
  assert.ok(bold,"**bold** should be a real safe bold element");
  assert.equal(bold.textContent,"Stockholm");
  assert.equal(created.find(n=>n.tag==="img"),undefined,"HTML injection must never create an image");
  assert.ok(created.some(n=>n.tag==="span" && n.textContent.includes("<img src=x")),"literal HTML should remain text");
  assert.ok(!created.some(n=>n.tag==="span" && n.textContent.includes("&#x20;")),"space entities should be cleaned");
});
