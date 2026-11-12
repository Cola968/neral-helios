"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const root=path.resolve(__dirname,"..");
function boot(events){
  let api=null;
  const response={ok:true,status:200,body:{getReader(){let i=0;return{async read(){return i<events.length?{done:false,value:Buffer.from(events[i++])}:{done:true}}}}}};
  const window={};const calls=[];
  const ctx={window,TextDecoder,fetch:async(url,opts)=>{calls.push({url,opts});return response;}};
  vm.runInNewContext(fs.readFileSync(path.join(root,"static/neral-web-live.js"),"utf8"),ctx);
  return {search:window.NeralWebLive.search,calls};
}
test("live search exposes contacted providers and checked sites before final result",async()=>{
  const packets=[
   '{"type":"provider","name":"DuckDuckGo HTML"}\n',
   '{"type":"source","source":{"title":"Berlin Airport","url":"https://berlin-airport.de/a","reachable":true}}\n',
   '{"type":"result","result":{"results":[{"url":"https://berlin-airport.de/a","reachable":true}],"providers_contacted":["DuckDuckGo HTML"]}}\n',
   '{"type":"end"}\n'];
  const b=boot(packets),events=[];
  const result=await b.search({query:'Suche BER',csrf:'valid',onEvent:e=>events.push(e)});
  assert.equal(events[0].type,'provider');assert.equal(events[1].type,'source');
  assert.equal(result.results.length,1);
  assert.equal(b.calls[0].url,'/api/web/search/stream');
  assert.equal(JSON.parse(b.calls[0].opts.body).verify_links,true);
});
test("the active status shows only real verified public domains and cleans up",()=>{
  const motion=fs.readFileSync(path.join(root,"static/neral-motion.js"),"utf8");
  assert.match(motion,/source-found/);assert.match(motion,/neral-live-sources/);
  assert.match(motion,/https:/);assert.match(motion,/source\?\.reachable!==true/);
  assert.match(motion,/provider/);
  const css=fs.readFileSync(path.join(root,"static/neral-reference-ui.css"),"utf8");
  assert.match(css,/\.neral-live-sources/);
  assert.match(css,/prefers-reduced-motion/);
});
test("the browser chat connects live web events to truthful search activity",()=>{
 const preview=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
 const html=fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
 assert.match(html,/neral-web-live\.js/);
 assert.match(preview,/NeralWebLive\?\.search/);
 assert.match(preview,/source-found/);
 assert.match(preview,/search-provider/);
});

test("live search shows only actual providers and checked domains beneath the status",()=>{
  class E{
    constructor(tag="div"){this.tagName=tag.toUpperCase();this.children=[];this.className="";this.textContent="";this.dataset={};this.parentNode=null;this.attributes={};}
    append(...nodes){for(const n of nodes){n.parentNode=this;this.children.push(n);}}
    remove(){if(!this.parentNode)return;this.parentNode.children=this.parentNode.children.filter(n=>n!==this);this.parentNode=null;}
    setAttribute(k,v){this.attributes[k]=String(v);}
    querySelector(s){for(const n of this.children){if(s.startsWith(".")&&n.className.split(" ").includes(s.slice(1)))return n;const v=n.querySelector(s);if(v)return v;}return null;}
  }
  const rootEl=new E("main"),listeners=new Map(),window={addEventListener:(n,cb)=>listeners.set(n,cb)};
  const document={getElementById:n=>n==="messages"?rootEl:null,createElement:n=>new E(n)};
  vm.runInNewContext(fs.readFileSync(path.join(root,"static/neral-motion.js"),"utf8"),
    {document,window,URL,Date,MutationObserver:class {}},{filename:"neral-motion.js"});
  const fire=(phase,extra={})=>listeners.get("neral:activity")({detail:{phase,mode:"think",...extra}});
  fire("searching");
  assert.equal(rootEl.querySelector(".neral-live-provider"),null,"never invent providers");
  fire("search-provider",{name:"DuckDuckGo HTML"});
  assert.match(rootEl.querySelector(".neral-live-provider").textContent,/DuckDuckGo/);
  fire("search-provider",{name:"Unknown provider"});
  assert.doesNotMatch(rootEl.querySelector(".neral-live-provider").textContent,/Unknown/);
  fire("source-found",{source:{title:"Private",url:"http://localhost/data",reachable:true}});
  fire("source-found",{source:{title:"Unchecked",url:"https://nasa.gov/data",reachable:false}});
  assert.equal(rootEl.querySelector(".neral-live-sites"),null,"unreachable links stay hidden");
  fire("source-found",{source:{title:"NASA mission",url:"https://nasa.gov/mission",reachable:true}});
  const row=rootEl.querySelector(".neral-live-sites").querySelector(".neral-live-site");
  assert.equal(row.querySelector(".neral-live-site-title").textContent,"NASA mission");
  assert.equal(row.querySelector(".neral-live-site-domain").textContent,"nasa.gov");
  assert.ok(row.querySelector(".neral-live-site-icon"),"Grok-style live row includes a real favicon");
  fire("thinking");
  assert.ok(rootEl.querySelector(".neral-live-sites"),"sources remain while model composes response");
  fire("done");
  assert.equal(rootEl.querySelector(".neral-activity"),null,"completed response has no stale working panel");
});

test("local ZIP includes live search script without cloud dependencies",()=>{
  const builder=fs.readFileSync(path.join(root,"scripts/build_neral_1_0_preview.py"),"utf8");
  assert.match(builder,/static\/neral-web-live\.js/);
  const script=fs.readFileSync(path.join(root,"static/neral-web-live.js"),"utf8");
  assert.doesNotMatch(script,/Math\.random|innerHTML|eval\(|setInterval\(/);
});

test("elapsed-time ticker is stopped on cancel; no fake progress or stale timers",()=>{
  class E{
    constructor(tag="div"){this.tagName=tag.toUpperCase();this.className="";this.dataset={};this.children=[];this.parentNode=null;this.textContent="";}
    append(...children){children.forEach(c=>{c.parentNode=this;this.children.push(c);});}
    remove(){if(!this.parentNode)return;this.parentNode.children=this.parentNode.children.filter(c=>c!==this);this.parentNode=null;}
    setAttribute(){}
    querySelector(sel){for(const child of this.children){if(child.className.split(" ").includes(sel.slice(1)))return child;const found=child.querySelector(sel);if(found)return found;}return null;}
  }
  let tick=null,stopped=0;
  const rootEl=new E("main"),handlers=new Map();
  const context={window:{addEventListener:(name,cb)=>handlers.set(name,cb)},
    document:{getElementById:id=>id==="messages"?rootEl:null,createElement:n=>new E(n)},
    Date,URL,MutationObserver:class{},setInterval:fn=>{tick=fn;return 5;},
    clearInterval:id=>{assert.equal(id,5);stopped++;}};
  vm.runInNewContext(fs.readFileSync(path.join(root,"static/neral-motion.js"),"utf8"),context);
  const fire=(phase,extra={})=>handlers.get("neral:activity")({detail:{phase,mode:"think",...extra}});
  fire("searching");assert.equal(typeof tick,"function");
  fire("cancelled");assert.equal(stopped,1);
  tick();assert.equal(rootEl.querySelector(".neral-activity"),null);
});


test("the live stream emits provider-discovered pages with icons before HTTPS checks",async()=>{
  const packets=[
    '{"type":"provider","name":"Wikipedia DE"}\n',
    '{"type":"candidate","source":{"title":"Flughafen Frankfurt Main","url":"https://de.wikipedia.org/wiki/Flughafen_Frankfurt_Main","discovered":true,"topically_relevant":true,"reachable":false}}\n',
    '{"type":"result","result":{"results":[],"previews":[{"title":"Flughafen Frankfurt Main","url":"https://de.wikipedia.org/wiki/Flughafen_Frankfurt_Main","discovered":true,"topically_relevant":true,"reachable":false}]}}\n',
    '{"type":"end"}\n'];
  const b=boot(packets),events=[];
  const result=await b.search({query:"Flughafen Frankfurt",csrf:"yes",onEvent:e=>events.push(e)});
  assert.equal(events[0].name,"Wikipedia DE");
  assert.equal(events[1].type,"candidate");
  assert.equal(events[1].source.reachable,false);
  assert.equal(result.previews.length,1);
  assert.equal(result.results.length,0);
  const motion=fs.readFileSync(path.join(root,"static/neral-motion.js"),"utf8");
  assert.match(motion,/"Wikipedia DE"/);
  assert.match(motion,/neral-live-site-icon/);
});
