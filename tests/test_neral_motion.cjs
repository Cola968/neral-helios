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
    get classList(){return {
      add:cls=>{if(!this.className.split(/\s+/).includes(cls))this.className+=" "+cls;},
      contains:cls=>this.className.split(/\s+/).includes(cls)
    };}
    get lastElementChild(){return this.children.at(-1)||null;}
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
      if(selector==="li")return this.tagName==="LI";
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
  const fire=(phase,mode="fast",extra={})=> {
    for(const fn of listeners.get("neral:activity")||[])fn({detail:{phase,mode,...extra}});
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
  assert.equal(activity(ui.preview)[0].querySelector(".neral-activity-label").textContent,"Sucht im Web");
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
  assert.match(source,/motionActivity\("searching"/);
  assert.match(source,/motionActivity\("checking"/);
  assert.match(source,/motionActivity\("thinking"/);
  assert.match(source,/motionActivity\("cancelled"/);
  assert.match(source,/motionActivity\("done"/);
  assert.match(source,/typeof CustomEvent/);
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

test("pure model generation never fabricates operational history",()=>{
  const ui=harness();
  ui.fire("thinking","think");
  assert.equal(ui.preview.querySelectorAll(".neral-activity-disclosure").length,0,
    "Model generation is not an auditable tool action");
  ui.fire("done");
  const answer=new ui.Element("div");answer.className="message assistant";
  ui.preview.append(answer);
  ui.fire("archive");
  assert.equal(answer.querySelectorAll(".neral-activity-history").length,0,
    "Do not invent a step list for an ordinary local response");
});
test("activity shows a calm NERAL mark and a real-state presence indicator",()=>{
  const ui=harness();
  ui.fire("searching","think");
  const row=activity(ui.preview)[0];
  assert.ok(row.querySelector(".neral-presence"));
  assert.equal(row.querySelectorAll(".neral-motion-glyph").length,1);
  assert.ok(row.querySelector(".neral-presence-signal"));
  assert.equal(row.querySelector(".neral-activity-label").textContent,"Sucht im Web");
  const css=fs.readFileSync(path.join(root,"static/neral-motion.css"),"utf8");
  assert.match(css,/\.neral-presence-signal/);
});

test("search history identifies real search results and excludes model reasoning",()=>{
  const ui=harness();
  ui.fire("searching","think");
  ui.fire("search-done","think",{count:0});
  ui.fire("thinking","think");
  const details=ui.preview.querySelector(".neral-activity-disclosure");
  assert.ok(details,"An actual attempted web search is observable");
  const rows=details.querySelectorAll("li");
  assert.equal(rows.length,1,"Model processing must not add a fake second step");
  assert.match(rows[0].textContent,/keine relevanten Treffer/);
  ui.fire("done");
  const answer=new ui.Element("div");answer.className="message assistant";
  ui.preview.append(answer);ui.fire("archive");
  const archive=answer.querySelector(".neral-activity-history");
  assert.ok(archive);
  assert.equal(archive.querySelectorAll("li").length,1);
});
test("attached files keep their own truthful status, without instantaneous fake stage changes",()=>{
  const source=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  assert.match(source,/if\(selectedAttachments\.length\)\s*motionActivity\("uploading"/);
  assert.match(source,/else\s*motionActivity\("thinking"/);
});
test("web searches require explicit search or freshness intent despite enabled toggle",()=>{
  const source=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  assert.match(source,/function shouldSearchWeb\(question\)/);
  assert.match(source,/\$\("useWebSearch"\)\.checked && !genericAttachmentOnly &&\s*shouldSearchWeb\(text\)/);
  assert.doesNotMatch(source,/\$\("useWebSearch"\)\.checked && !skipWebForLocalBasics\(text\)/);
});

test("search intent uses actual question rather than opt-in alone",()=>{
  const source=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const block=source.slice(source.indexOf("  function skipWebForLocalBasics"),
    source.indexOf("  function relevantSearchResult"));
  const probe=vm.runInNewContext(block+"\nshouldSearchWeb", {});
  assert.equal(probe("Erkläre mir ein schwieriges Physikthema"),false);
  assert.equal(probe("Hallo, wie geht es dir?"),false);
  assert.equal(probe("Suche aktuelle Meldungen zum BER"),true);
  assert.equal(probe("Welche Nachrichten gibt es heute?"),true);
  assert.equal(probe("Was ist die Hauptstadt von Deutschland?"),false);
});

test("invalid web queries are rejected before announcing a search",()=>{
  const source=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const gate=source.indexOf("if(text.length>160");
  const event=source.indexOf('motionActivity("searching",mode)');
  const request=source.indexOf('const search=await json("/api/web/search"');
  assert.ok(gate>=0 && gate<event && event<request);
});

test("sources must match topical words, never generic search or recency terms",()=>{
  const source=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const block=source.slice(source.indexOf("  function relevantSearchResult"),
    source.indexOf("  function appendSources"));
  const relevant=vm.runInNewContext(block+"\nrelevantSearchResult",{});
  assert.equal(relevant("Suche aktuelle Meldungen zum BER",{
    title:"Aktuelle Meldungen zur Rechtschreibung",
    snippet:"Neueste News und Meldungen",url:"https://example.org/duden"
  }),false,"A generic 'aktuell' overlap is not topical evidence");
  assert.equal(relevant("Suche aktuelle Meldungen zum BER",{
    title:"BER Flughafen: Neue Flugverbindungen",
    snippet:"Aktuelle Entwicklungen am Berlin Brandenburg Airport",
    url:"https://example.org/ber"
  }),true,"BER itself must be searchable despite its three-letter acronym");
  assert.equal(relevant("Neue BER Flughafen Flugverbindungen",{
    title:"BER Flughafen: Neue Strecken",snippet:"Die Flugverbindungen wachsen.",
    url:"https://example.org/ber"
  }),true);
  assert.equal(relevant("Neue BER Flughafen Flugverbindungen",{
    title:"BER Parkplatz",snippet:"Parken am Flughafen.",
    url:"https://example.org/parking"
  }),false,"One topical keyword should not pass a multi-topic query");
});
