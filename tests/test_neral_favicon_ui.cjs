"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const root=path.resolve(__dirname,"..");const read=p=>fs.readFileSync(path.join(root,p),"utf8");
class El {
 constructor(tag="div"){this.tagName=tag.toUpperCase();this.children=[];this.className="";this.dataset={};this.textContent="";this.parentNode=null;this.handlers={};this.attrs={};}
 append(...nodes){for(const node of nodes){this.children.push(node);node.parentNode=this;}}
 prepend(node){this.children.unshift(node);node.parentNode=this;}
 setAttribute(k,v){this.attrs[k]=String(v);}
 addEventListener(k,v){this.handlers[k]=v;}
 remove(){if(this.parentNode){this.parentNode.children=this.parentNode.children.filter(c=>c!==this);this.parentNode=null;}}
 get classList(){return {add:name=>{this.className+=" "+name;},contains:name=>this.className.split(/\s+/).includes(name)}};
 querySelector(s){for(const c of this.children){if(s.startsWith(".")&&c.className.split(/\s+/).includes(s.slice(1)))return c;if(s===c.tagName.toLowerCase())return c;const nested=c.querySelector(s);if(nested)return nested;}return null;}
}
test("finished answer uses real favicon images in stacked sources and source list",()=>{
 const src=read("static/neral-preview.js"),a=src.indexOf("  function appendSources("),b=src.indexOf("  async function json(",a);
 const render=vm.runInNewContext(src.slice(a,b)+"\nappendSources",{URL,document:{createElement:t=>new El(t)}});
 const actions=new El(),answer={querySelector:k=>k===".message-actions"?actions:null};
 render(answer,[{title:"GitHub NERAL",url:"https://github.com/Cola968/neral-helios",reachable:true},
                 {title:"NASA mission",url:"https://www.nasa.gov/mission",reachable:true}]);
 const stack=actions.querySelector(".neral-source-stack");assert.ok(stack);
 const markers=stack.children;assert.equal(markers.length,2);
 const icons=markers.map(m=>m.querySelector(".neral-source-favicon"));
 assert.ok(icons.every(Boolean),"Every shown source needs its real site icon");
 assert.equal(icons[0].src,"/api/web/favicon?host=github.com");
 assert.equal(icons[1].src,"/api/web/favicon?host=www.nasa.gov");
 assert.ok(!icons[0].src.startsWith("https://"),"Icons never bypass same-origin CSP");
 assert.ok(actions.querySelector(".neral-source-list-icon"),"Sources menu has the icons too");
 icons[0].handlers.error?.();assert.equal(markers[0].querySelector(".neral-source-favicon"),null,
  "Browser icon failures use the neutral placeholder");
});
test("live search puts the site favicon beside the real verified domain",()=>{
 const listeners=new Map(),area=new El("main");
 const ctx={window:{addEventListener:(name,fn)=>listeners.set(name,fn)},
  document:{getElementById:id=>id==="messages"?area:null,createElement:t=>new El(t)},
  URL,Date,MutationObserver:class {}};
 vm.runInNewContext(read("static/neral-motion.js"),ctx,{timeout:1500});
 const fire=(phase,extra={})=>listeners.get("neral:activity")({detail:{phase,mode:"think",...extra}});
 fire("searching");fire("source-found",{source:{title:"GitHub",url:"https://github.com/Cola968/neral-helios",reachable:true}});
 const link=area.querySelector(".neral-live-site");assert.ok(link);
 const icon=link.querySelector(".neral-live-site-icon");assert.ok(icon);
 assert.equal(icon.src,"/api/web/favicon?host=github.com");
 assert.equal(icon.attrs.alt,"");
 fire("done");assert.equal(area.querySelector(".neral-live-site"),null);
});
test("site favicon CSS gives real 16px icons, fallback and reduced motion",()=>{
 const css=read("static/neral-reference-ui.css");
 assert.match(css,/\.neral-source-favicon/);
 assert.match(css,/\.neral-source-list-icon/);
 assert.match(css,/\.neral-live-site-icon/);
 assert.match(css,/neral-favicon-loaded/);
 assert.match(css,/prefers-reduced-motion/);
});

test("source cards expose numbered entries and truthful search snippets",()=>{
 const src=read("static/neral-preview.js"),a=src.indexOf("  function appendSources("),b=src.indexOf("  async function json(",a);
 const render=vm.runInNewContext(src.slice(a,b)+"\nappendSources",{URL,document:{createElement:t=>new El(t)}});
 const actions=new El(),answer={querySelector:k=>k===".message-actions"?actions:null};
 render(answer,[{title:"Flughafen Berlin Brandenburg",url:"https://berlin-airport.de/de",reachable:true,
                 snippet:"Terminals und Anreise zum Flughafen"}],true);
 const card=actions.querySelector(".neral-source-link");
 assert.ok(card);
 assert.ok(card.querySelector(".neral-source-index"),"Each card identifies its search-result number");
 assert.match(card.querySelector(".neral-source-snippet").textContent,/Terminals und Anreise/);
 assert.match(actions.querySelector(".neral-source-notice").textContent,/nicht.*geprüft/i);
});
test("empty live web results show an explicit, accessible search-status note",()=>{
 const src=read("static/neral-preview.js"),a=src.indexOf("  function appendSources("),b=src.indexOf("  async function json(",a);
 const render=vm.runInNewContext(src.slice(a,b)+"\nappendSources",{URL,document:{createElement:t=>new El(t)}});
 const actions=new El(),answer={querySelector:k=>k===".message-actions"?actions:null};
 render(answer,[],true);
 assert.match(actions.querySelector(".neral-source-empty").textContent,/keine.*Treffer/i);
 assert.equal(actions.querySelector(".neral-source-link"),null);
});
test("source cards use brief hover and entry motion, respecting reduced motion",()=>{
 const css=read("static/neral-reference-ui.css");
 assert.match(css,/@keyframes neral-source-appear/);
 assert.match(css,/\.neral-source-link:hover/);
 assert.match(css,/prefers-reduced-motion:reduce[\s\S]*neral-source-link/);
});
