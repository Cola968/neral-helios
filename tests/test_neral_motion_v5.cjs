"use strict";
const {test}=require("node:test"), assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const root=path.resolve(__dirname,"..");
const read=p=>fs.readFileSync(path.join(root,p),"utf8");
class El {
  constructor(tag="div") {this.tagName=tag.toUpperCase();this.children=[];this.className="";this.dataset={};this.textContent="";this.attrs={};}
  append(...children){this.children.push(...children);}
  prepend(child){this.children.unshift(child);}
  setAttribute(key,value){this.attrs[key]=String(value);}
  querySelector(query){for(const child of this.children){if(query.startsWith(".")&&child.className.split(" ").includes(query.slice(1)))return child;if(query===child.tagName.toLowerCase())return child;let c=child.querySelector(query);if(c)return c;}return null;}
}
function sourceRenderer(){let source=read("static/neral-preview.js");let body=source.slice(source.indexOf("  function appendSources("),source.indexOf("  async function json("));return vm.runInNewContext(body+"\nappendSources",{URL,document:{createElement:t=>new El(t)}});}
test("sources appear inside their own answer action bar, not as orphan chat messages",()=>{
  const render=sourceRenderer(),actions=new El("div");actions.className="message-actions";
  const answer={querySelector:selector=>selector===".message-actions"?actions:null};
  render(answer,[{title:"BER Flughafen",url:"https://www.berlin-airport.de/de.html",reachable:true}]);
  const details=actions.querySelector(".neral-source-disclosure");
  assert.ok(details,"compact chip must live in answer actions");
  assert.equal(details.querySelector(".neral-source-label").textContent,"1 Quelle");
  const link=details.querySelector("a");assert.ok(link);
  assert.equal(link.attrs.rel,"noopener noreferrer");
  assert.equal(link.attrs.target,"_blank");
});
test("unsafe, unverified and off-topic sources never produce a citation chip",()=>{
  const render=sourceRenderer(),actions=new El("div");actions.className="message-actions";
  const answer={querySelector:()=>actions};
  render(answer,[{title:"Malicious",url:"http://localhost:3000/",reachable:true},
    {title:"Unverified",url:"https://example.org/news",reachable:false}]);
  assert.equal(actions.children.length,0);
});
test("operational status never produces a large expandable activity box or historical timeline",()=>{
  const js=read("static/neral-motion.js");
  assert.doesNotMatch(js,/createHistory\(|neral-activity-disclosure|neral-activity-history/);
  assert.match(js,/neral-activity-label/);
  const css=read("static/neral-motion.css");
  assert.match(css,/\.neral-activity\{/);
  assert.match(css,/prefers-reduced-motion/);
});
test("only confirmed live search results may be injected or displayed",()=>{
  const js=read("static/neral-preview.js");
  assert.match(js,/verify_links:\s*true/);
  assert.match(js,/hit\.reachable===true/);
  assert.match(js,/appendSources\(answerNode,sources\)/);
  assert.doesNotMatch(js,/if\(sources\.length\)appendSources\(sources\)/);
});

function searchFunctions(){
  const source=read("static/neral-preview.js");
  const start=source.indexOf("  function skipWebForLocalBasics(");
  const end=source.indexOf("  function appendSources(",start);
  assert.ok(start>=0&&end>start);
  return vm.runInNewContext(source.slice(start,end)+
    "\n({shouldSearchWeb,makeSearchQuery,webEvidencePrompt,normaliseSearchRefusal})",{});
}
test("checked Web searches short factual follow-ups and reuses only a previous web query",()=>{
  const x=searchFunctions();
  assert.equal(x.shouldSearchWeb("warum nicht"),false);
  assert.equal(x.shouldSearchWeb("warum nicht","Turbofan-Triebwerk"),true);
  assert.equal(x.shouldSearchWeb("Wie funktioniert ein Turbofan?"),true);
  assert.equal(x.shouldSearchWeb("Hallo"),false);
  assert.equal(x.makeSearchQuery("warum nicht","Turbofan-Triebwerk"),"Turbofan-Triebwerk warum nicht");
  assert.equal(x.makeSearchQuery("Wie funktioniert ein Turbofan?","geheimes Passwort"),"Wie funktioniert ein Turbofan?");
});
test("a completed search never lets NERAL claim it cannot perform web search",()=>{
  const x=searchFunctions();
  const refusal="Ich kann im Moment keine Web-Suche durchführen, da ich keine Echtzeitinformationen habe.";
  assert.match(x.webEvidencePrompt("Was gibt es Neues?",[],true),/Websuche.{0,100}ausgeführt/i);
  assert.match(x.webEvidencePrompt("Was gibt es Neues?",[],true),/keine.{0,40}Treffer/i);
  assert.doesNotMatch(x.webEvidencePrompt("Was gibt es Neues?",[],true),/bis 2024/i);
  assert.match(x.normaliseSearchRefusal(refusal,2,true),/2.{0,40}Quellen|2.{0,40}Treffer/i);
  assert.equal(x.normaliseSearchRefusal(refusal,0,false),refusal);
  assert.equal(x.normaliseSearchRefusal("Ein Turbofan erzeugt Schub.",2,true),"Ein Turbofan erzeugt Schub.");
});
test("every reachable source is immediately visible with its own real favicon request",()=>{
  const render=sourceRenderer(),actions=new El("div");actions.className="message-actions";
  render({querySelector:()=>actions},["a","b","c","d"].map((x,i)=>({
    title:"Quelle "+i,url:"https://"+x+".example.org/article",reachable:true
  })));
  const details=actions.querySelector(".neral-source-disclosure");
  assert.equal(details.open,true,"source list must not be collapsed by default");
  function collect(el,tag){return [...(el.tagName===tag.toUpperCase()?[el]:[]),
    ...el.children.flatMap(x=>collect(x,tag))];}
  const links=collect(details,"a"),icons=collect(details,"img");
  assert.equal(links.length,4);
  for(const link of links){
    const host=new URL(link.href).hostname;
    const ownIcon=collect(link,"img");
    assert.equal(ownIcon.length,1);
    assert.equal(ownIcon[0].src,"/api/web/favicon?host="+encodeURIComponent(host));
  }
  assert.ok(icons.length>=4);
});
test("web activity displays the actual submitted query without fabricated image-search steps",()=>{
  const js=read("static/neral-motion.js");
  assert.match(js,/search-query/);
  assert.match(js,/neral-live-search-query/);
  assert.doesNotMatch(js,/Bildersuche tur[b]?ofan|Ausgeführte 3 Suchen/);
});

test("conversation clearing and switching invalidate the previous web query",()=>{
  const js=read("static/neral-preview.js");
  assert.match(js,/addEventListener\?\.\("neral:chat-cleared",\(\)=>\{lastWebQuery="";\}\)/);
  assert.match(js,/addEventListener\?\.\("neral:load-chat",\(\)=>\{lastWebQuery="";\}\)/);
  assert.match(js,/if\(!\$\("useWebSearch"\)\.checked\)lastWebQuery="";/);
});
