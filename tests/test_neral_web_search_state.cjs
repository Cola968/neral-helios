"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const {TextEncoder}=require("node:util");

async function runPrompt(question,{web=false,results=[],previews=[]}={}){
  const handlers=new Map(),elements={},calls=[];
  function element(id){
    return elements[id] ||= {value:"",textContent:"",checked:false,disabled:false,children:[],
      addEventListener(kind,fn){handlers.set(id+":"+kind,fn);},
      querySelector(){return id==="chatForm"?{value:"fast"}:null;},
      append(...x){this.children.push(...x);},replaceChildren(...x){this.children=x;},
      remove(){},focus(){},setAttribute(){},getAttribute(){return null;},
      scrollHeight:0,scrollTop:0};
  }
  const turns=Array.isArray(question)?question:[question];
  element("prompt").value=turns[0];
  element("useWebSearch").checked=web;
  async function fetch(url,options={}){
    calls.push({url,body:options.body&&typeof options.body==="string"?JSON.parse(options.body):null});
    if(url.endsWith("/session"))
      return {ok:true,json:async()=>({csrf:"TEST_CSRF",model:"qwen3:8b"})};
    if(url.endsWith("/status"))
      return {ok:true,json:async()=>({model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false})};
    if(url==="/api/web/search")
      return {ok:true,json:async()=>({results,previews,provider:"fixture"})};
    if(url==="/api/neral/chat/generate")
      return {ok:true,json:async()=>({text:"Antwort",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false,model_generated:true})};
    throw new Error("Unexpected URL "+url);
  }
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8"),{
    document:{getElementById:element,createElement:tag=>({tag,textContent:"",children:[],className:"",
      append(...x){this.children.push(...x)},remove(){},setAttribute(){},getAttribute(){return null}})},
    fetch,TextEncoder,AbortSignal,AbortController,Error,console,
  },{timeout:2000});
  await new Promise(resolve=>setTimeout(resolve,10));
  for(const [i,turn] of turns.entries()){
    if(i>0)element("prompt").value=turn;
    await handlers.get("chatForm:submit")({preventDefault(){}});
  }
  return {calls,elements};
}

test("explicit internet request with Web disabled shows a useful message and never calls model",async()=>{
  const {calls,elements}=await runPrompt("Suche im Internet nach aktuellen Raumfahrtmissionen",{web:false});
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,0);
  assert.equal(calls.filter(c=>c.url==="/api/neral/chat/generate").length,0);
  assert.match(elements.error.textContent,/Web.*aktivier|aktivier.*Web/i);
  assert.equal(elements.prompt.value,"Suche im Internet nach aktuellen Raumfahrtmissionen");
});

test("search with zero relevant sources informs model that a search actually ran",async()=>{
  const question="Suche im Internet nach aktuellen Raumfahrtmissionen";
  const {calls}=await runPrompt(question,{web:true,results:[]});
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,1);
  const modelCall=calls.find(c=>c.url==="/api/neral/chat/generate");
  assert.ok(modelCall,"the local model still receives the question");
  assert.match(modelCall.body.prompt,/NERAL-WEBSTATUS/);
  assert.match(modelCall.body.prompt,/keine passenden Treffer/);
  assert.match(modelCall.body.prompt,/URSPRÜNGLICHE FRAGE/);
});

test("plain imperative Suche nach ... requires Web consent, not a model refusal", async()=>{
  const question="suche nach dem berliner flughafen";
  const {calls,elements}=await runPrompt(question,{web:false});
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,0);
  assert.equal(calls.filter(c=>c.url==="/api/neral/chat/generate").length,0);
  assert.match(elements.error.textContent,/Web.*aktivier|aktivier.*Web/i);
  assert.equal(elements.prompt.value,question);
});

test("stale Qwen refusal is replaced only after a real search, including 'externe Websuche'",()=>{
  const src=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  const start=src.indexOf("  function normaliseSearchRefusal(");
  const end=src.indexOf("  function relevantSearchResult(",start);
  const clean=vm.runInNewContext(src.slice(start,end)+"\nnormaliseSearchRefusal");
  const first="Ich habe eine interne Suche durchgeführt. Da keine externe Websuche oder Dateizugriff in diesem Chat aktiv ist, basiert diese Antwort auf meinem trainierten Wissen.";
  const second="Ich kann keine externe Websuche durchführen oder aktuelle Daten aus dem Internet abrufen, da dafür notwendige Werkzeuge nicht verfügbar sind.";
  for(const sentence of [first,second]){
    assert.doesNotMatch(clean(sentence,2,true),/keine externe Websuche/i);
    assert.match(clean(sentence,2,true),/2 erreichbare Treffer/);
    assert.equal(clean(sentence,2,false),sentence,
      "Never rewrite a genuine offline response without a completed web search");
  }
  assert.match(clean(second,0,true),/keine passenden erreichbaren Treffer/i);
});


test("explicit German search submits the subject, not the imperative sentence",()=>{
  const src=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  const start=src.indexOf("  function isSearchFollowup(");
  const end=src.indexOf("  // Snippets are untrusted evidence",start);
  assert.ok(start>0&&end>start);
  const getQuery=vm.runInNewContext(src.slice(start,end)+"\nmakeSearchQuery");
  assert.equal(getQuery("suche nach grok ai"),"grok ai");
  assert.equal(getQuery("Suche nach dem Berliner Flughafen"),"dem Berliner Flughafen");
  assert.equal(getQuery("Bitte suche nach dem Wetter in Berlin"),"dem Wetter in Berlin");
  assert.equal(getQuery("warum?", "grok ai"),"grok ai warum?");
  assert.equal(getQuery("Welche Flugzeuge nutzt Condor?"),"Flugzeuge nutzt Condor");
});


test("verified web hits override stale Qwen empty-result boilerplate from the user's screenshot",()=>{
  const source=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  const start=source.indexOf("  function normaliseSearchRefusal(");
  const end=source.indexOf("  function relevantSearchResult(",start);
  const normalise=vm.runInNewContext(source.slice(start,end)+"\nnormaliseSearchRefusal");
  const stale="Ich habe eine Websuche nach „Grok AI“ durchgeführt. Es liegen keine belastbaren Treffer vor, die ich in diesem Chat bestätigen kann. Falls Sie weitere Informationen suchen, kann ich helfen.";
  const corrected=normalise(stale,5,true);
  assert.match(corrected,/5 erreichbare Treffer|5 passende Webseiten/i);
  assert.doesNotMatch(corrected,/keine belastbaren Treffer/i);
  assert.equal(normalise(stale,0,true),stale,"Zero source results must not be turned into success");
  assert.equal(normalise(stale,5,false),stale,"Offline answers must not be tampered with");
});


test("failed HEAD probes cannot hide genuine relevant search snippets from the local answer",async()=>{
  const hits=[{title:"Frankfurt Airport – Fraport",url:"https://www.frankfurt-airport.com/en.html",
    snippet:"Information about Frankfurt Airport",discovered:true,topically_relevant:true,reachable:false}];
  const {calls}=await runPrompt("suche nach Flughafen Frankfurt",
    {web:true,results:[],previews:hits});
  const search=calls.find(c=>c.url==="/api/web/search");
  assert.equal(search.body.query,"Flughafen Frankfurt");
  const model=calls.find(c=>c.url==="/api/neral/chat/generate");
  assert.equal(model,undefined,"Provider snippets must not wait for local LLM");
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,1);
});


test("web hits render directly without a model wait and include a web tool badge",async()=>{
  const hits=[{title:"EZB Leitzins aktuell",url:"https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html",
    snippet:"Aktuelle EZB Zinssätze und Termine",discovered:true,topically_relevant:true,reachable:false}];
  const {calls,elements}=await runPrompt(
    "Suche bitte aktuell im Web: Wie hoch ist heute der Leitzins der EZB? Nenne mir deine Quellen.",
    {web:true,results:[],previews:hits});
  assert.equal(calls.filter(c=>c.url==="/api/neral/chat/generate").length,0);
  const rendered=JSON.stringify(elements.messages.children);
  assert.match(rendered,/EZB Leitzins aktuell/);
  assert.match(rendered,/Werkzeug: Websuche \(1 Quelle\)/);
  assert.doesNotMatch(rendered,/Grenze:/);
});

test("Frankfurt no-hits boilerplate is corrected when independent web tool has actual sources",()=>{
  const source=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  const start=source.indexOf("  function normaliseSearchRefusal(");
  const end=source.indexOf("  function relevantSearchResult(",start);
  const clean=vm.runInNewContext(source.slice(start,end)+"\nnormaliseSearchRefusal");
  const text="Ich habe die Websuche für deine Anfrage zum Flughafen Frankfurt durchgeführt. "+
    "Wie in der Statusmeldung angegeben, lieferte diese Suche keine passenden oder belastbaren Treffer. "+
    "Falls du allgemeine Informationen suchst, kann ich dir helfen.";
  const result=clean(text,5,true,5);
  assert.match(result,/5 erreichbare Treffer/);
  assert.doesNotMatch(result,/keine passenden oder belastbaren Treffer/);
  assert.equal(clean(text,0,true,0),text);
  assert.equal(clean(text,5,false,5),text);
});


test("subjectless web command with greeting asks for a topic, never searches the word Hallo",async()=>{
  const {calls,elements}=await runPrompt("hallo suche im internet",{web:true});
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,0);
  assert.equal(calls.filter(c=>c.url==="/api/neral/chat/generate").length,0);
  const rendered=JSON.stringify(elements.messages.children);
  assert.match(rendered,/Wonach.*Internet suchen/i);
  assert.doesNotMatch(rendered,/NERAL-WEBSTATUS|5 Quellen/);
});

test("extract search topics after greeting and treat entity corrections as topical follow-ups",()=>{
  const source=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  const start=source.indexOf("  function skipWebForLocalBasics(");
  const end=source.indexOf("  // Snippets are untrusted evidence",start);
  const get=vm.runInNewContext(source.slice(start,end)+"\n({makeSearchQuery,shouldSearchWeb})");
  assert.equal(get.makeSearchQuery("Hallo suche im Internet"),"");
  assert.equal(get.makeSearchQuery("Hallo, suche im Internet nach PEAK Spiel speichern"),
               "PEAK Spiel speichern");
  assert.equal(get.makeSearchQuery("^suche im internet ob man in PEAK speichern kann"),
               "ob PEAK speichern");
  assert.equal(get.shouldSearchWeb("nochmal PEAK ist ein Computerspiel","PEAK speichern"),true);
  const correction=get.makeSearchQuery("nochmal PEAK ist ein Computerspiel","PEAK speichern");
  assert.match(correction,/PEAK/);
  assert.match(correction,/Computerspiel/);
  assert.equal(get.shouldSearchWeb("Hallo 😊","PEAK speichern"),false);
});


test("topic-less explicit web search continues safely after the user supplies only a topic",async()=>{
  const {calls,elements}=await runPrompt(
    ["Hallo, suche im Internet","PEAK Computerspiel Spielstand speichern"],{web:true});
  const searches=calls.filter(c=>c.url==="/api/web/search");
  assert.equal(searches.length,1);
  assert.equal(searches[0].body.query,"PEAK Computerspiel Spielstand speichern");
  assert.equal(calls.filter(c=>c.url==="/api/neral/chat/generate").length,1);
  assert.equal(elements.error.textContent,"");
});

test("a cancelled subject request does not search private small talk or later turns",async()=>{
  const {calls}=await runPrompt(
    ["Hallo suche im Internet","Nein danke","Ich möchte nur ein bisschen reden."],{web:true});
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,0);
});
