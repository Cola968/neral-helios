"use strict";
// Race regressions for sensitive user inputs in NERAL 1.0 Preview.
const test=require("node:test");
const assert=require("node:assert/strict");
const vm=require("node:vm");
const fs=require("node:fs");
const path=require("node:path");
const {TextEncoder}=require("node:util");

const deferred=()=>{
  let resolve,reject;
  const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});
  return {promise,resolve,reject};
};
const tick=()=>new Promise(done=>setTimeout(done,10));
function harness({deferResponse=false}={}) {
  const listeners=new Map(),calls=[],elements=Object.create(null),recognitions=[];
  let activeResponse=null;
  const element=id=>elements[id] ||= {
    value:"",checked:false,disabled:false,files:[],textContent:"",
    addEventListener(type,fn){listeners.set(id+":"+type,fn);},
    querySelector(){return null;},
    appended:[],
    append(...args){this.appended.push(...args);},
    replaceChildren(){this.appended=[];},
    remove(){},focus(){},setAttribute(){},
    scrollHeight:0,scrollTop:0,
    requestSubmit(){return listeners.get("chatForm:submit")({preventDefault(){}})}
  };
  element("chatForm").querySelector=()=>({value:"fast"});
  element("voiceAutoSend").checked=true;
  class Recognition{
    constructor(){this.started=false;this.aborted=false;this.stopped=false;recognitions.push(this);}
    start(){this.started=true;}
    stop(){this.stopped=true;}
    abort(){this.aborted=true;}
  }
  const fetch=async(url,options={})=>{
    calls.push({url,options});
    if(url.endsWith("/session"))return {ok:true,json:async()=>({csrf:"FAKE_SESSION_TOKEN",model:"qwen3:8b"})};
    if(url.endsWith("/status"))return {ok:true,json:async()=>({model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false})};
    if(url==="/api/neral/chat/generate"){
      if(deferResponse){
        activeResponse=deferred();
        await activeResponse.promise;
      }
      return {ok:true,json:async()=>({text:"SPÄTE MODELLANTWORT",model_generated:true,
        model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false})};
    }
    if(url==="/api/web/search")return {ok:true,json:async()=>({results:[
      {title:"Example",url:"https://docs.python.org/3/",snippet:"Official docs"}]})};
    throw Error("Unexpected request "+url);
  };
  const source=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  vm.runInNewContext(source,{
    document:{getElementById:element,createElement:tag=>({
      tag,textContent:"",className:"",children:[],
      append(...items){this.children.push(...items)},remove(){},setAttribute(){}
    })},
    fetch,TextEncoder,AbortSignal,AbortController,Error,console,
    window:{SpeechRecognition:Recognition,speechSynthesis:{cancel(){},speak(){}}},
    SpeechSynthesisUtterance:class {constructor(text){this.text=text;}},
  },{timeout:2000});
  return {element,listeners,calls,recognitions,response:()=>activeResponse};
}
const send=x=>x.listeners.get("chatForm:submit")({preventDefault(){}});
const select=(x,files)=>{
  x.element("projectFiles").files=files;
  return x.listeners.get("projectFiles:change")();
};
const file=(name,body)=>({name,size:new TextEncoder().encode(body).length,text:async()=>body});

test("rapid file replacement cannot restore earlier sensitive content",async()=>{
  const h=harness();await tick();
  const old=deferred();
  const first=select(h,[{name:"secret.txt",size:10,text:()=>old.promise}]);
  await select(h,[file("public.txt","SAFE_REPLACEMENT")]);
  old.resolve("PRIVATE_OLD");
  await first;
  assert.match(h.element("fileStatus").textContent,/public\.txt/);
  assert.doesNotMatch(h.element("fileStatus").textContent,/secret\.txt/);
  h.element("prompt").value="Fasse die Datei zusammen.";
  await send(h);
  const sent=h.calls.find(c=>c.url==="/api/neral/chat/generate");
  assert.ok(sent);
  const body=JSON.parse(sent.options.body);
  assert.match(body.prompt,/SAFE_REPLACEMENT/);
  assert.doesNotMatch(body.prompt,/PRIVATE_OLD|secret\.txt/);
});

test("clear during an outstanding read invalidates that file permanently",async()=>{
  const h=harness();await tick();
  const old=deferred();
  const loading=select(h,[{name:"private.txt",size:12,text:()=>old.promise}]);
  h.listeners.get("clear:click")();
  old.resolve("NEVER_SEND_ME");
  await loading;
  assert.match(h.element("fileStatus").textContent,/Keine Dateien/);
  h.element("prompt").value="Neue Nachricht";
  await send(h);
  const sent=h.calls.find(c=>c.url==="/api/neral/chat/generate");
  assert.ok(sent);
  assert.equal(JSON.parse(sent.options.body).prompt,"Neue Nachricht");
});

test("stopping or clearing microphone ignores late browser result events",async()=>{
  const h=harness();await tick();
  const click=h.listeners.get("voiceInput:click");
  click();
  const previous=h.recognitions[0];
  click(); // user explicitly cancels
  assert.equal(previous.aborted,true,"Cancellation must abort, not finalize recognition");
  previous.onresult({results:[[{transcript:"IGNORED_OLD_VOICE"}]]});
  await tick();
  assert.equal(h.calls.filter(c=>c.url==="/api/neral/chat/generate").length,0);
  assert.notEqual(h.element("prompt").value,"IGNORED_OLD_VOICE");
  click();
  const current=h.recognitions[1];
  h.listeners.get("clear:click")();
  assert.equal(current.aborted,true);
  current.onresult({results:[[{transcript:"IGNORED_AFTER_CLEAR"}]]});
  await tick();
  assert.equal(h.calls.filter(c=>c.url==="/api/neral/chat/generate").length,0);
});

test("clear while a model request is active prevents stale reply entering new conversation",async()=>{
  const h=harness({deferResponse:true});await tick();
  h.element("prompt").value="Alte Frage";
  const pending=send(h);
  await tick();
  assert.ok(h.response(),"Model call was started");
  h.listeners.get("clear:click")();
  h.response().resolve();
  await pending;
  const rendered=h.element("messages").appended.map(item=>item.children?.map(child=>child.textContent||"").join(" ")||"").join(" ");
  assert.doesNotMatch(rendered,/SPÄTE MODELLANTWORT/);
  assert.match(rendered,/gelöscht/);
});
