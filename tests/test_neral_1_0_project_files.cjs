"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const {TextEncoder}=require("node:util");

function harness(reply={}) {
  const handlers=new Map(),calls=[],elements={},created=[],clipboard=[];
  function makeElement(id) {
    const ownHandlers=new Map(),attrs=new Map();
    return {
      id, tagName:id, value:"", checked:false,disabled:false,
      files:[],textContent:"",title:"",className:"",children:[],
      addEventListener(type,fn){ownHandlers.set(type,fn);handlers.set(id+":"+type,fn);},
      click(){const fn=ownHandlers.get("click");return fn&&fn();},
      setAttribute(k,v){attrs.set(k,v);},
      getAttribute(k){return attrs.get(k)||null;},
      querySelector(){return null;},
      append(...items){this.children.push(...items);},
      replaceChildren(){this.children=[];},
      remove(){},focus(){},scrollHeight:0,scrollTop:0,
      requestSubmit(){return handlers.get("chatForm:submit")({preventDefault(){}})}
    };
  }
  const element=id=>elements[id] ||= makeElement(id);
  element("chatForm").querySelector=()=>({value:"fast"});
  class FormDataMock {
    constructor(){this.values=[];}
    append(key,value,name){this.values.push({key,value,name});}
    get(key){return this.values.find(x=>x.key===key)?.value;}
    getAll(key){return this.values.filter(x=>x.key===key).map(x=>x.value);}
  }
  const fetch=async(url,opts={})=>{
    calls.push({url,opts});
    if(url.endsWith("/session"))return {ok:true,json:async()=>({csrf:"abc",model:"qwen3:8b"})};
    if(url.endsWith("/status"))return {ok:true,json:async()=>({
      model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false
    })};
    if(url==="/api/neral/chat/generate" ||
       url==="/api/neral/preview/attachment-chat")
      return {ok:true,json:async()=>({
        text:"Auswertung: 42",model_origin:"local_foundation_not_neral_adapter",
        adapter_loaded:false,vision_models:["qwen3-vl:2b"],...reply
      })};
    throw Error("Unexpected URL "+url);
  };
  const document={
    getElementById:element,
    createElement:tag=>{
      const el=makeElement("created-"+created.length);
      el.tagName=tag;
      created.push(el);return el;
    }
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8"),{
    document,window:{},fetch,TextEncoder,AbortSignal,FormData:FormDataMock,
    navigator:{clipboard:{writeText:async text=>{clipboard.push(text);}}},
    AbortController,Error,console
  },{timeout:2000});
  return {element,handlers,calls,created,clipboard};
}
const flush=()=>new Promise(r=>setTimeout(r,10));

test("chosen image/document bytes go to localhost multipart only after submit",async()=>{
  const x=harness();await flush();
  const source={name:"photo.png",size:103,readAsText(){throw Error("do not read");}};
  x.element("projectFiles").files=[source];
  await x.handlers.get("projectFiles:change")();
  assert.equal(x.calls.filter(y=>y.url.includes("attachment-chat")).length,0);
  assert.match(x.element("fileStatus").textContent,/photo.png/);
  x.element("prompt").value="Was sieht man?";
  await x.handlers.get("chatForm:submit")({preventDefault(){}});
  const sent=x.calls.find(y=>y.url==="/api/neral/preview/attachment-chat");
  assert.ok(sent);
  assert.equal(sent.opts.body.get("prompt"),"Was sieht man?");
  assert.equal(sent.opts.body.get("mode"),"generate");
  assert.equal(sent.opts.body.getAll("files")[0],source);
  assert.equal(sent.opts.headers["X-Neral-CSRF"],"abc");
  assert.equal(sent.opts.headers["Content-Type"],undefined,
    "Browser must supply multipart boundary, never manual Content-Type");
  assert.match(x.element("fileStatus").textContent,/Keine Dateien/);
});

test("untrusted extension or size rejected without reading; ordinary chat preserved",async()=>{
  const x=harness();await flush();
  x.element("projectFiles").files=[{name:"malware.exe",size:20000,
                                    text:async()=>{throw Error("must not read");}}];
  await x.handlers.get("projectFiles:change")();
  assert.match(x.element("fileStatus").textContent,/abgelehnt/i);
  x.element("prompt").value="Guten Tag";
  await x.handlers.get("chatForm:submit")({preventDefault(){}});
  const sent=x.calls.find(y=>y.url==="/api/neral/chat/generate");
  assert.equal(JSON.parse(sent.opts.body).prompt,"Guten Tag");
  assert.equal(x.calls.filter(y=>y.url.includes("attachment-chat")).length,0);
});

test("assistant responses have functional like toggle and copy without feedback upload",async()=>{
  const x=harness();await flush();
  x.element("prompt").value="Bitte antworten.";
  await x.handlers.get("chatForm:submit")({preventDefault(){}});
  const like=x.created.find(e=>e.textContent==="♡ Gefällt mir");
  const copy=x.created.find(e=>e.textContent==="Kopieren");
  assert.ok(like);assert.ok(copy);
  assert.equal(like.getAttribute("aria-pressed"),"false");
  like.click();
  assert.equal(like.getAttribute("aria-pressed"),"true");
  like.click();
  assert.equal(like.getAttribute("aria-pressed"),"false");
  await copy.click();
  assert.deepEqual(x.clipboard,["Auswertung: 42"]);
  assert.equal(copy.textContent,"Kopiert");
  assert.equal(x.calls.filter(c=>/feedback|like|copy/.test(c.url)).length,0);
});

test("truncated attachment excerpts are disclosed beside the model answer",async()=>{
  const x=harness({vision_models:[],
    attachments:[{name:"long.txt",kind:"document",truncated:true}]});
  await flush();
  const source={name:"long.txt",size:10};
  x.element("projectFiles").files=[source];
  x.handlers.get("projectFiles:change")();
  x.element("prompt").value="Bitte zusammenfassen.";
  await x.handlers.get("chatForm:submit")({preventDefault(){}});
  const assistantHeaders=x.created.filter(e=>e.tagName==="strong")
    .map(e=>e.textContent);
  assert.ok(assistantHeaders.some(t=>/Auszüge gekürzt/i.test(t)),
    "Users must see that not all document content was processed");
});

test("image-only request with web toggle never searches for generic attachment placeholder",async()=>{
  const x=harness();await flush();
  x.element("useWebSearch").checked=true;
  x.element("projectFiles").files=[{name:"screenshot.png",size:100}];
  x.handlers.get("projectFiles:change")();
  x.element("prompt").value="";
  await x.handlers.get("chatForm:submit")({preventDefault(){}});
  assert.equal(x.calls.filter(c=>c.url==="/api/web/search").length,0,
    "The phrase 'Analysiere die Anhänge' must not be leaked as a web query");
  const upload=x.calls.find(c=>c.url==="/api/neral/preview/attachment-chat");
  assert.ok(upload,"real image bytes must still follow multipart path");
  assert.equal(upload.opts.body.get("prompt"),"Analysiere die Anhänge.");
});

test("answer provenance displays selected HIGH foundation and does not claim files were understood",async()=>{
  const x=harness({model:"qwen3.5:9b",attachment_context_verified:false});
  await flush();
  x.element("chatForm").querySelector=()=>({value:"think"});
  x.element("projectFiles").files=[{name:"scan.png",size:200}];
  x.handlers.get("projectFiles:change")();
  x.element("prompt").value="Was steht auf diesem Bild?";
  await x.handlers.get("chatForm:submit")({preventDefault(){}});
  const labels=x.created.filter(e=>e.tagName==="strong").map(e=>e.textContent);
  const heading=labels.find(s=>s.includes("NERAL HIGH"));
  assert.ok(heading,"HIGH response is visible");
  assert.match(heading,/qwen3\.5:9b/,"show real model identity, not an implicit Qwen8B label");
  assert.match(heading,/Anhänge verarbeitet/);
  assert.doesNotMatch(heading,/Anhänge ausgewertet/,"upload alone is not proof of content understanding");
});
