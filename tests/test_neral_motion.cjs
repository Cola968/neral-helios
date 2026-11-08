// Minimal DOM regression tests for honest local NERAL UI motion.
// No network, GPU, or real model required.
"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const root=path.resolve(__dirname,"..");

function harness(){
  const observers=[];
  const listeners=new Map();
  class Element {
    constructor(tag="div"){
      this.tagName=tag.toUpperCase(); this.children=[]; this.dataset={};
      this.attributes={}; this.className=""; this.textContent="";
      this.parentNode=null; this.scrollTop=0; this.scrollHeight=100; this.events=new Map();
    }
    addEventListener(type,fn){this.events.set(type,fn);}
    click(){this.events.get("click")?.({type:"click"});}
    get classList(){return {add:cls=>{if(!this.className.split(/\s+/).includes(cls))this.className+=" "+cls;}};}
    append(...nodes){
      for(const node of nodes){node.parentNode=this;this.children.push(node);}
      this.notify("childList");
    }
    prepend(node){node.parentNode=this;this.children.unshift(node);this.notify("childList");}
    remove(){
      if(!this.parentNode)return;
      const p=this.parentNode; p.children=p.children.filter(n=>n!==this);
      this.parentNode=null;p.notify("childList");
    }
    insertAdjacentElement(place,node){
      assert.equal(place,"afterend");
      const parent=this.parentNode;
      const idx=parent.children.indexOf(this);
      node.parentNode=parent;parent.children.splice(idx+1,0,node);
      parent.notify("childList");
    }
    setAttribute(key,value){this.attributes[key]=String(value);this.notify("attributes",key);}
    getAttribute(key){return this.attributes[key]??null;}
    notify(type,key){
      for(const ob of observers){
        if(ob.target!==this||!ob.options[type])continue;
        if(type==="attributes" && ob.options.attributeFilter?.length &&
           !ob.options.attributeFilter.includes(key))continue;
        ob.callback([]);
      }
    }
    matches(selector){
      if(selector===".message.pending")return this.className.split(/\s+/).includes("message") && this.className.split(/\s+/).includes("pending");
      if(selector.startsWith("."))return this.className.split(/\s+/).includes(selector.slice(1));
      return false;
    }
    querySelectorAll(selector){
      const found=[];
      function walk(el){for(const child of el.children){if(child.matches(selector))found.push(child);walk(child);}}
      walk(this);return found;
    }
    querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  }
  const preview=new Element();
  const log=new Element();
  const area=new Element();
  const transcript=new Element();
  area.append(transcript);
  const thinkSwitch={checked:false};
  const stop=new Element("button");
  const comparePanel=new Element();
  area.append(comparePanel);
  const elements={
    messages:preview,log,neralLocalChatAnswer:transcript,
    neralLocalChatThink:thinkSwitch,neralComparePanel:comparePanel,stop
  };
  class MutationObserver{
    constructor(callback){this.callback=callback;}
    observe(target,options){this.target=target;this.options=options;observers.push(this);}
    disconnect(){const i=observers.indexOf(this);if(i>=0)observers.splice(i,1);}
  }
  const window={
    addEventListener(type,callback){
      if(!listeners.has(type))listeners.set(type,[]);
      listeners.get(type).push(callback);
    }
  };
  const document={
    getElementById(id){return elements[id]||null;},
    createElement(tag){return new Element(tag);}
  };
  const script=fs.readFileSync(path.join(root,"static/neral-motion.js"),"utf8");
  vm.runInNewContext(script,{window,document,MutationObserver},{filename:"neral-motion.js"});
  const fire=(phase,mode="fast")=> {
    for(const fn of listeners.get("neral:activity")||[])fn({detail:{phase,mode}});
  };
  return {preview,log,transcript,area,thinkSwitch,comparePanel,stop,Element,fire};
}

function activity(container){
  return container.querySelectorAll(".neral-activity");
}
test("thinking/searching lifecycle is accurate, idempotent and cancellable",()=>{
  const ui=harness();
  ui.fire("thinking","think");
  assert.equal(activity(ui.preview).length,1);
  assert.equal(activity(ui.preview)[0].querySelector(".neral-activity-label").textContent,"Denkt nach");
  ui.fire("thinking","think");
  assert.equal(activity(ui.preview).length,1,"duplicate phases must not duplicate indicators");
  ui.fire("searching","think");
  assert.equal(activity(ui.preview)[0].querySelector(".neral-activity-label").textContent,"Durchsucht das Web");
  ui.fire("checking");
  assert.equal(activity(ui.preview)[0].querySelector(".neral-activity-label").textContent,"Prüft Python-Syntax");
  ui.fire("done");
  assert.equal(activity(ui.preview).length,0);
  ui.fire("thinking","fast");
  assert.equal(activity(ui.preview)[0].querySelector(".neral-activity-label").textContent,"Verarbeitet Anfrage");
  ui.fire("cancelled");
  assert.equal(activity(ui.preview).length,0,"abort must remove the indicator immediately");
});
test("Helios pending status clears glyph on first actual streaming text",()=>{
  const ui=harness();
  const pending=new ui.Element("article");
  pending.className="message assistant pending";
  ui.log.append(pending);
  assert.equal(pending.querySelectorAll(".neral-pending-glyph").length,1);
  // The streamer sets data-partial. The matching CSS hides the glyph instantly.
  const css=fs.readFileSync(path.join(root,"static/neral-motion.css"),"utf8");
  assert.match(css,/\.message\.pending\[data-partial\]\s+\.neral-pending-glyph\s*\{[^}]*display:\s*none/);
  pending.dataset.partial="actual model output";
  ui.log.notify("childList");
  assert.equal(pending.querySelectorAll(".neral-pending-glyph").length,1);
  ui.log.append(new ui.Element("p"));
  assert.equal(pending.querySelectorAll(".neral-pending-glyph").length,1);
});
test("Stop immediately freezes pending animation without altering streamed text",()=>{
  const ui=harness();
  const pending=new ui.Element("article");
  pending.className="message assistant pending";
  const content=new ui.Element("div");content.className="content";content.textContent="Actual partial reply";
  pending.dataset.partial="Actual partial reply";
  pending.append(content);
  ui.log.append(pending);
  ui.stop.click();
  assert.match(pending.className,/neral-cancel-requested/);
  assert.equal(content.textContent,"Actual partial reply","stop must preserve existing partial text");
  const css=fs.readFileSync(path.join(root,"static/neral-motion.css"),"utf8");
  assert.match(css,/\.message\.pending\.neral-cancel-requested\[data-partial\] \.content::after/);
});
test("embedded local model activity follows actual aria-busy, without fake percentages",()=>{
  const ui=harness();
  ui.thinkSwitch.checked=true;
  ui.transcript.setAttribute("aria-busy","true");
  assert.equal(activity(ui.area).length,1);
  assert.equal(activity(ui.area)[0].querySelector(".neral-activity-label").textContent,"Denkt nach");
  ui.transcript.removeAttribute?.("aria-busy");
  // Fake DOM has no removeAttribute: drive the real transition via explicit false.
  ui.transcript.setAttribute("aria-busy","false");
  assert.equal(activity(ui.area).length,0);
});
test("motion contract keeps text stable, reduced motion and keyboard-visible controls",()=>{
  const css=fs.readFileSync(path.join(root,"static/neral-motion.css"),"utf8");
  const js=fs.readFileSync(path.join(root,"static/neral-motion.js"),"utf8");
  assert.match(css,/@media\s*\(prefers-reduced-motion:\s*reduce\)/);
  assert.match(css,/\.message\.pending\[data-partial\]\s+\.content::after/);
  assert.match(css,/#modelPicker\[open\]/);
  assert.match(css,/\.sidebar\.open/);
  assert.match(css,/dialog\[open\]/);
  assert.match(css,/#voiceInput\[aria-pressed="true"\]/);
  assert.match(css,/#neralComparePanel\[aria-busy="true"\]/);
  assert.doesNotMatch(js,/Math\.random|setInterval|innerHTML|token.*setTimeout/);
  assert.match(js,/aria-live/);
});
test("local preview announces real stages and removes indicator on abort and finally",()=>{
  const source=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const html=fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
  const index=fs.readFileSync(path.join(root,"templates/index.html"),"utf8");
  assert.match(source,/phase:"searching"/);
  assert.match(source,/phase:"checking"/);
  assert.match(source,/phase:"thinking"/);
  assert.match(source,/phase:"cancelled"/);
  assert.match(source,/phase:"done"/);
  assert.match(html,/neral-motion\.css/);
  assert.match(html,/neral-motion\.js/);
  assert.match(index,/neral-motion\.css/);
  assert.match(index,/neral-motion\.js/);
  assert.match(index,/src="\/static\/icon.svg"/,"approved NERAL logo must remain");
  const sw=fs.readFileSync(path.join(root,"static/sw.js"),"utf8");
  assert.match(sw,/neral-shell-v4/);
  assert.match(sw,/\/static\/neral-motion\.css/);
  assert.match(sw,/\/static\/neral-motion\.js/);
});
