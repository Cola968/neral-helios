"use strict";
// Regression contracts for NERAL's real operational activity timeline.
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
const root=path.resolve(__dirname,"..");

function boot(){
  const listeners=new Map();
  class El{
    constructor(tag="div"){
      this.tagName=tag.toUpperCase();this.children=[];this.parentNode=null;
      this.className="";this.dataset={};this.textContent="";this.attrs={};
      this.scrollHeight=200;this.scrollTop=0;
    }
    append(...nodes){for(const node of nodes){node.parentNode=this;this.children.push(node);}}
    remove(){if(this.parentNode)this.parentNode.children=this.parentNode.children.filter(x=>x!==this);this.parentNode=null;}
    setAttribute(k,v){this.attrs[k]=String(v);}
    getAttribute(k){return this.attrs[k]??null;}
    get lastElementChild(){return this.children.at(-1)||null;}
    get classList(){return {contains:x=>this.className.split(" ").includes(x)};}
    querySelector(q){
      const matches=n=>q.startsWith(".")&&n.className.split(" ").includes(q.slice(1));
      for(const el of this.children){
        if(matches(el))return el;
        const child=el.querySelector(q);
        if(child)return child;
      }
      return null;
    }
  }
  const preview=new El("div");
  const doc={getElementById:id=>id==="messages"?preview:null,createElement:tag=>new El(tag)};
  const win={addEventListener:(k,fn)=>{listeners.set(k,fn)}};
  const code=fs.readFileSync(path.join(root,"static/neral-motion.js"),"utf8");
  vm.runInNewContext(code,{window:win,document:doc,MutationObserver:class {}});
  return {
    preview,El,
    fire:(phase,extra={})=>{
      const fn=listeners.get("neral:activity");
      assert.equal(typeof fn,"function");
      fn({detail:{phase,mode:"think",...extra}});
    }
  };
}

test("only real lifecycle phases produce entries and an expandable report",()=>{
  const ui=boot();
  ui.fire("searching");
  ui.fire("search-done",{count:3});
  ui.fire("uploading",{count:2});
  ui.fire("thinking");
  ui.fire("done");
  const answer=new ui.El("div");answer.className="message assistant";
  ui.preview.append(answer);
  ui.fire("archive");
  const details=answer.querySelector(".neral-activity-history");
  assert.ok(details,"completed answer should own a collapsed activity list");
  assert.equal(details.tagName,"DETAILS");
  assert.notEqual(details.getAttribute("open"),"true","history is collapsed by default");
  const history=details.querySelector(".neral-activity-history-list");
  assert.ok(history);
  assert.equal(history.children.length,2,"only web search and attachment request are recorded; model work is not a tool");
  assert.match(history.textContent+" "+history.children.map(x=>x.textContent).join(" "),/3 relevante Treffer/);
  assert.equal(ui.preview.querySelector(".neral-activity"),null,"no waiting indicator after completion");
});
test("expanded live activity remains open as real tool states change",()=>{
  const ui=boot();
  ui.fire("searching");
  const disclosure=ui.preview.querySelector(".neral-activity-disclosure");
  assert.ok(disclosure);
  disclosure.open=true;
  ui.fire("search-done",{count:2});
  const refreshed=ui.preview.querySelector(".neral-activity-disclosure");
  assert.ok(refreshed.open, "live updates must not collapse a panel the user expanded");
});
test("cancelled work never gets a fake success history",()=>{
  const ui=boot();
  ui.fire("thinking");
  ui.fire("cancelled");
  const answer=new ui.El("div");answer.className="message assistant";
  ui.preview.append(answer);
  ui.fire("archive");
  assert.equal(answer.querySelector(".neral-activity-history"),null);
});
test("no fake search step is implied when only the local model runs",()=>{
  const ui=boot();
  ui.fire("thinking");
  ui.fire("done");
  const answer=new ui.El("div");answer.className="message assistant";
  ui.preview.append(answer);
  ui.fire("archive");
  const details=answer.querySelector(".neral-activity-history");
  assert.equal(details,null,"model-only turns must not fabricate an activity timeline");
});
test("attachment cards are user-initiated and text-only",()=>{
  const src=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const html=fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
  const css=fs.readFileSync(path.join(root,"static/neral-motion.css"),"utf8");
  assert.match(html,/id="attachmentChips"/);
  assert.match(src,/function renderAttachmentChips/);
  assert.match(src,/motionActivity\("uploading"/);
  assert.match(src,/motionActivity\("search-done"/);
  assert.match(src,/motionActivity\("archive"/);
  assert.match(css, /prefers-reduced-motion:reduce/);
  assert.doesNotMatch(src,/innerHTML\s*=/);
});
