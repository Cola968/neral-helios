"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const root=path.resolve(__dirname,"..");
const read=p=>fs.readFileSync(path.join(root,p),"utf8");
const dataReply="| Stadt | Fahrgäste |\n| --- | ---: |\n| Berlin | 25 |\n| Hamburg | 10 |";
function harness(reply=dataReply){
  const handlers=new Map(),elements=new Map(),calls=[];
  class El{
    constructor(tag="div"){
      this.tagName=tag.toUpperCase();this.id="";this.value="";this.textContent="";
      this.children=[];this.attrs={};this.className="";this.dataset={};
      this.disabled=false;this.hidden=false;this.style={};
      this.scrollHeight=150;this.scrollTop=0;
    }
    append(...nodes){this.children.push(...nodes);}
    replaceChildren(...nodes){this.children=[...nodes];}
    setAttribute(k,v){this.attrs[k]=String(v);}
    removeAttribute(k){delete this.attrs[k];}
    getAttribute(k){return this.attrs[k]??null;}
    addEventListener(kind,fn){handlers.set((this.id||"unnamed")+":"+kind,fn);}
    focus(){}
  }
  function element(id){
    if(!elements.has(id)){const node=new El("div");node.id=id;elements.set(id,node);}
    return elements.get(id);
  }
  const document={getElementById:element,createElement:tag=>new El(tag)};
  const window={};
  const fetch=async(url,options)=>{
    if(url.endsWith("/status")){
      return {ok:true,json:async()=>({model:"qwen3:8b",
        model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false})};
    }
    assert.match(url,/^\/api\/neral\/chat\/generate$/);
    calls.push({url,options});
    return {ok:true,json:async()=>({text:reply,model_generated:true,
       model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false})};
  };
  const context={document,window,fetch,TextEncoder,AbortSignal,Intl,Math,Number,console};
  vm.runInNewContext(read("static/neral-dynamic-ui.js"),context,{timeout:1500});
  vm.runInNewContext(read("static/neral-local-chat.js"),context,{timeout:1500});
  element("neralLocalChatPrompt").value="Zeige Tabelle";
  const submit=handlers.get("neralLocalChatForm:submit");
  assert.equal(typeof submit,"function","real local-chat submit handler is bound");
  return {El,element,submit:()=>submit({preventDefault(){}}),calls};
}
function flatten(node){
  return [node.textContent,...node.children.map(flatten)].join(" ");
}
function findByClass(root,className){
  if(root.className.split(" ").includes(className))return root;
  for(const child of root.children){const found=findByClass(child,className);if(found)return found;}
  return null;
}
test("actual embedded NERAL AI chat mounts a table below a completed reply",async()=>{
  const h=harness();
  await h.submit();
  assert.equal(h.calls.length,1);
  const content=h.element("neralLocalChatAnswer");
  assert.equal(content.children.length,2,"user and assistant are retained");
  assert.match(flatten(content),/Berlin/);
  const widget=findByClass(content.children[1],"neral-dynamic-widget");
  assert.ok(widget,"actual reply must contain interactive UI widget");
  assert.match(flatten(widget),/Interaktive Tabelle/);
  assert.match(flatten(widget),/ungeprüft/);
});
test("plain answers never add a decorative widget",async()=>{
  const h=harness("Eine normale Antwort ohne strukturierte Daten.");
  await h.submit();
  assert.equal(findByClass(h.element("neralLocalChatAnswer"),"neral-dynamic-widget"),null);
});
test("local answer remains visible even if rendering fails or plan is absent",async()=>{
  const h=harness("Einfacher Text");
  await h.submit();
  assert.match(flatten(h.element("neralLocalChatAnswer")),/Einfacher Text/);
});
test("Helios main chat and embedded chat load the same vetted UI assets",()=>{
  const html=read("templates/index.html");
  const source=read("static/neral.js");
  const embedded=read("static/neral-local-chat.js");
  for(const asset of ["neral-dynamic-ui.css","neral-dynamic-ui.js"]){
    assert.ok(html.includes("/static/"+asset),asset+" missing in Helios");
  }
  assert.match(source,/window\.NeralDynamicUI/);
  assert.match(source,/\.mount\(/);
  assert.match(embedded,/NeralDynamicUI/);
  assert.match(embedded,/\.mount\(/);
  assert.match(source,/current\.messages\.entries\(\)/);
});
test("Windows EXE root and ZIP both serve a Dynamic UI enabled chat",()=>{
  const server=read("neral_preview_server.py");
  const desktop=read("neral_oneclick.py");
  const preview=read("templates/neral_preview.html");
  const pack=read("scripts/build_neral_1_0_preview.py");
  assert.match(desktop,/create_preview_app/);
  assert.match(server,/render_template\("neral_preview\.html"/);
  assert.ok(preview.includes("/static/neral-dynamic-ui.js"));
  assert.ok(pack.includes('"static/neral-dynamic-ui.js"'));
});
