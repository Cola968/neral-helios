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
