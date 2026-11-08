"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const {TextEncoder}=require("node:util");

function harness({ignoreAbort=false}={}){
  const handlers=new Map(), elements={}, calls=[], rendered=[];
  const pending=[];
  const windowEvents=new Map();
  const browser={
    addEventListener(name,callback){windowEvents.set(name,callback);},
    dispatchEvent(event){const fn=windowEvents.get(event.type);if(fn)fn(event);}
  };
  class CustomEvent {constructor(type,options={}){this.type=type;this.detail=options.detail;}}
  function element(id){
    return elements[id] ||= {
      value:"", checked:false, disabled:false, files:[],
      textContent:"", title:"", children:[],
      addEventListener(name,fn){handlers.set(id+":"+name,fn);},
      querySelector(){if(id==="chatForm")return {value:"fast"};return null;},
      append(...x){this.children.push(...x);if(id==="messages")rendered.push(...x);},
      replaceChildren(){this.children=[];if(id==="messages")rendered.length=0;},
      remove(){},focus(){},setAttribute(){},
      scrollHeight:0,scrollTop:0
    };
  }
  element("prompt").value="Eine kurze Frage";
  const fetch=async(url,opts={})=>{
    if(url.endsWith("/session"))return {ok:true,json:async()=>({csrf:"abc",model:"qwen3:8b"})};
    if(url.endsWith("/status"))return {ok:true,json:async()=>({
      model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",
      adapter_loaded:false
    })};
    calls.push({url,opts,body:JSON.parse(opts.body)});
    return new Promise((resolve,reject)=>{
      const task={url,opts,resolve,reject};
      pending.push(task);
      if(!ignoreAbort){
        opts.signal.addEventListener("abort",()=>reject(Object.assign(new Error("Cancelled"),{name:"AbortError"})),{once:true});
      }
    });
  };
  const source=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  vm.runInNewContext(source,{
    window:browser,CustomEvent,document:{getElementById:element,createElement:tag=>({
      tagName:tag.toUpperCase(),textContent:"",className:"",children:[],
      append(...x){this.children.push(...x);},remove(){}
    })},
    fetch,TextEncoder,AbortSignal,AbortController,Error,console
  },{timeout:2000});
  return {element,elements,handlers,rendered,calls,pending,
          stop:()=>handlers.get("stopRequest:click")(),
          clear:()=>handlers.get("clear:click")(),
          submit:()=>handlers.get("chatForm:submit")({preventDefault(){}}),
          emit:(name,detail)=>browser.dispatchEvent(new CustomEvent(name,{detail})),
          flush:()=>new Promise(r=>setImmediate(r))};
}
const modelResult=text=>({ok:true,json:async()=>({
  text,model_generated:true,
  model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false
})});

test("Stop aborts only the browser request and preserves input for retry",async()=>{
  const html=fs.readFileSync(path.join(__dirname,"../templates/neral_preview.html"),"utf8");
  assert.match(html,/id="stopRequest"/);
  const h=harness();await h.flush();
  assert.equal(h.elements.stopRequest.disabled,true);
  const req=h.submit();
  assert.equal(h.pending.length,1);
  assert.equal(h.elements.stopRequest.disabled,false);
  h.stop();await req;
  assert.equal(h.calls[0].opts.signal.aborted,true);
  assert.equal(h.rendered.length,1,"Submitted user message remains after Stop");
  assert.ok(h.rendered[0].className.includes("user"));
  assert.equal(h.elements.prompt.value,"Eine kurze Frage");
  assert.equal(h.elements.stopRequest.disabled,true);
  assert.equal(h.elements.submit.disabled,false);
  assert.match(h.elements.error.textContent,/Ollama|Berechnung|weiterlaufen/i);
});

test("clear cannot resurrect a late model reply, leak old history, or trigger speech",async()=>{
  const h=harness({ignoreAbort:true});await h.flush();
  const req=h.submit();assert.equal(h.pending.length,1);
  h.clear();
  h.pending[0].resolve(modelResult("SHOULD_NOT_RESURRECT"));
  await req;
  assert.equal(h.calls[0].opts.signal.aborted,true);
  assert.ok(!JSON.stringify(h.rendered).includes("SHOULD_NOT_RESURRECT"));
  h.elements.prompt.value="Neue Frage";
  const next=h.submit();
  assert.equal(h.calls[1].body.history.length,0);
  h.pending[1].resolve(modelResult("Neue Antwort"));
  await next;
  assert.ok(JSON.stringify(h.rendered).includes("Neue Antwort"));
});

test("clear during an opt-in WEB SEARCH cannot launch a zombie model request",async()=>{
  const h=harness({ignoreAbort:true});await h.flush();
  h.element("useWebSearch").checked=true;
  h.element("prompt").value="Suche Python-Dokumentation";
  const active=h.submit();
  assert.equal(h.pending[0].url,"/api/web/search");
  h.clear();
  h.pending[0].resolve({ok:true,json:async()=>({results:[
    {title:"Docs",snippet:"Example",url:"https://docs.python.org/3/"}
  ]})});
  await active;
  assert.equal(h.calls.filter(x=>x.url==="/api/neral/chat/generate").length,0);
  assert.equal(h.elements.stopRequest.disabled,true);
});


test("account switch clears an unsent private draft and selected files", async()=>{
  const h=harness();await h.flush();
  h.element("prompt").value="Privater Entwurf von Konto A";
  h.element("projectFiles").value="C:\\fakepath\\private.txt";
  h.element("projectFiles").files=[{name:"private.txt",size:20}];
  h.handlers.get("projectFiles:change")();
  assert.match(h.element("fileStatus").textContent,/private\.txt/);
  h.emit("neral:load-chat",{messages:[]});
  assert.equal(h.element("prompt").value,"");
  assert.equal(h.element("projectFiles").value,"");
  assert.match(h.element("fileStatus").textContent,/Keine Dateien/);
});
