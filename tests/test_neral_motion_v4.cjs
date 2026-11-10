"use strict";
// Motion V4 contracts: real state, stable content, no fabricated progress or motion.
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const root=path.resolve(__dirname,".."),read=p=>fs.readFileSync(path.join(root,p),"utf8");
function boot(){
  const listeners=new Map();
  class Element{
    constructor(tag="div"){
      this.tagName=tag.toUpperCase();this.className="";this.children=[];
      this.dataset={};this.textContent="";this.attributes={};
      this.parentNode=null;this.scrollHeight=120;this.scrollTop=0;
    }
    append(...nodes){for(const n of nodes){n.parentNode=this;this.children.push(n);}}
    setAttribute(name,value){this.attributes[name]=String(value);}
    getAttribute(name){return this.attributes[name]||null;}
    get lastElementChild(){return this.children.at(-1)||null;}
    remove(){if(this.parentNode){
      this.parentNode.children=this.parentNode.children.filter(n=>n!==this);
      this.parentNode=null;
    }}
    get classList(){return {contains:n=>this.className.split(" ").includes(n)};}
    querySelector(q){
      if(!q.startsWith("."))return null;
      for(const child of this.children){
        if(child.className.split(" ").includes(q.slice(1)))return child;
        const found=child.querySelector(q);if(found)return found;
      }
      return null;
    }
  }
  const area=new Element("main");
  const window={addEventListener:(name,fn)=>listeners.set(name,fn)};
  const document={getElementById:id=>id==="messages"?area:null,createElement:tag=>new Element(tag)};
  vm.runInNewContext(read("static/neral-motion.js"),
    {window,document,MutationObserver:class {}},{timeout:1500});
  function fire(phase,extra={}){
    const cb=listeners.get("neral:activity");
    assert.equal(typeof cb,"function");
    cb({detail:{phase,mode:"think",...extra}});
  }
  return {area,fire};
}
test("working state uses one tiny phase-specific symbol, never a logo card",()=>{
  const ui=boot();
  ui.fire("thinking");
  const activity=ui.area.querySelector(".neral-activity");
  assert.ok(activity);
  assert.ok(activity.querySelector(".neral-status-symbol"));
  assert.equal(activity.querySelector(".neral-motion-glyph"),null);
  assert.equal(activity.querySelector(".neral-presence"),null);
  assert.equal(activity.dataset.phase,"thinking");
  ui.fire("searching");
  assert.equal(activity.dataset.phase,"searching");
  ui.fire("uploading");
  assert.equal(activity.dataset.phase,"uploading");
  ui.fire("cancelled");
  assert.equal(ui.area.querySelector(".neral-activity"),null);
});
test("logo never pulses or rotates endlessly; reduced motion is comprehensive",()=>{
  const css=read("static/neral-motion.css");
  assert.match(css,/\.neral-presence\s*\{/);
  assert.match(css,/\.neral-presence-signal\s*\{/);
  assert.match(css,/\[data-phase="thinking"\]\s+\.neral-presence-signal/);
  assert.match(css,/\[data-phase="searching"\]\s+\.neral-presence-signal/);
  assert.match(css,/\[data-phase="uploading"\]\s+\.neral-presence-signal/);
  assert.match(css,/\.neral-motion-glyph,.neral-pending-glyph\s*\{[^}]*animation:\s*none/s);
  assert.match(css,/@media\s*\(prefers-reduced-motion:\s*reduce\)/);
  assert.match(css,/\.neral-presence-signal[^\n]*animation:\s*none!important/);
  assert.doesNotMatch(css,/filter:\s*blur\(/);
});
test("new replies enter once, restored history never replays entrance",()=>{
  const preview=read("static/neral-preview.js"),css=read("static/neral-motion.css");
  assert.match(preview,/function append\(role,text,origin,animate=false\)/);
  assert.match(preview,/neral-motion-enter/);
  assert.match(preview,/append\(turn\.role,turn\.content\)/);
  assert.match(preview,/append\("assistant",result\.text,origin,true\)/);
  assert.match(css,/#messages>\.message\.neral-motion-enter/);
  assert.doesNotMatch(css,/#messages>\.message\s*\{animation:neral-arrive/);
});
test("failed or cancelled requests never announce successful completion",()=>{
  const preview=read("static/neral-preview.js");
  assert.match(preview,/let completedTurn=false/);
  assert.match(preview,/if\(!completedTurn\)\s*motionActivity\(/);
  assert.match(preview,/checkResponded\s*\?\s*"done"\s*:\s*"error"/);
  assert.doesNotMatch(preview,/finally\s*\{\s*motionActivity\("done"/);
});
test("widgets animate their real data only and native controls remain accessible",()=>{
  const css=read("static/neral-dynamic-ui.css"),motion=read("static/neral-motion.css");
  assert.match(css,/\.neral-motion-enter\s+\.neral-dynamic-widget/);
  assert.match(css,/@media\s*\(prefers-reduced-motion:\s*reduce\)/);
  assert.match(css,/\.neral-dui-fill[^\n]*animation:\s*none!important/);
  assert.match(motion,/\.neral-activity-history\[open\]\s+\.neral-activity-history-list/);
  assert.match(motion,/\.neral-activity-disclosure\[open\]\s+\.neral-activity-history-list/);
  const js=read("static/neral-motion.js");
  assert.doesNotMatch(js,/Math\.random|innerHTML|new Function/);
  assert.match(js,/Date\.now\(\)/, "Display time must use real elapsed wall-clock time");
});
