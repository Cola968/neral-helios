"use strict";
const test=require("node:test"),assert=require("node:assert/strict"),vm=require("node:vm");
const fs=require("node:fs"),path=require("node:path");
const {TextEncoder}=require("node:util");

async function exercise(promptText,results){
  const handlers=new Map(),elements={},calls=[];
  function element(id){
    return elements[id] ||= {
      value:"",textContent:"",disabled:false,checked:false,
      addEventListener(kind,callback){handlers.set(id+":"+kind,callback);},
      querySelector(){return id==="chatForm"?{value:"fast"}:null;},append(){},replaceChildren(){},
      remove(){},focus(){},setAttribute(){},getAttribute(){return null;},
      scrollHeight:0,scrollTop:0,
    };
  }
  element("prompt").value=promptText;
  element("useWebSearch").checked=true;
  const fetch=async(url,options={})=>{
    calls.push({url,body:options.body?JSON.parse(options.body):null});
    if(url.endsWith("/session"))
      return {ok:true,json:async()=>({csrf:"TEST_CSRF",model:"qwen3:8b"})};
    if(url.endsWith("/status"))
      return {ok:true,json:async()=>({
        model:"qwen3:8b",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false
      })};
    if(url==="/api/web/search")
      return {ok:true,json:async()=>({results,provider:"fixture"})};
    if(url==="/api/neral/chat/generate")
      return {ok:true,json:async()=>({
        text:"Antwort",model_origin:"local_foundation_not_neral_adapter",
        adapter_loaded:false,model_generated:true
      })};
    throw Error("unexpected URL "+url);
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8"),{
    document:{getElementById:element,createElement:tag=>({
      tag,textContent:"",append(){},remove(){},className:"",setAttribute(){},
    })},fetch,TextEncoder,AbortSignal,AbortController,Error,console,
  },{timeout:2000});
  await new Promise(resolve=>setTimeout(resolve,5));
  await handlers.get("chatForm:submit")({preventDefault(){}});
  return calls;
}

test("greetings and identity do not upload a web search query even with toggle enabled", async()=>{
  for(const question of ["hallo","Wer bist du?","Nein, wie heißt du, welches Modell bist du?"]){
    const calls=await exercise(question,[]);
    assert.equal(calls.filter(c=>c.url==="/api/web/search").length,0,question);
    assert.equal(calls.filter(c=>c.url==="/api/neral/chat/generate").length,1,question);
  }
});

test("irrelevant search hits never contaminate model prompt",async()=>{
  const calls=await exercise("Suche: Ist PW1000 ein Turbofan oder Turboprop?",[
    {title:"ChatGPT jailbreak prompt",url:"https://github.com/example/chatgpt",snippet:"LLM prompt engineering"}
  ]);
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,1);
  const body=calls.find(c=>c.url==="/api/neral/chat/generate").body;
  assert.match(body.prompt,/NERAL-WEBSTATUS/);
  assert.match(body.prompt,/keine passenden Treffer/);
  assert.ok(body.prompt.endsWith("Suche: Ist PW1000 ein Turbofan oder Turboprop?"));
  assert.doesNotMatch(body.prompt,/ChatGPT jailbreak/);
});

test("electric fields follow-up rejects unrelated TV links",async()=>{
  const calls=await exercise("Suche elektrische Felder und Stromkreise",[
    {title:"Fernsehprogramm heute abend 20:15 Uhr",url:"https://tvspielfilm.de/tv-programm",snippet:"Fernsehen heute"},
    {title:"Das VOX TV-Programm von heute",url:"https://tvspielfilm.de/vox",snippet:"TV-Sendungen"}
  ]);
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,1);
  const body=calls.find(c=>c.url==="/api/neral/chat/generate").body;
  assert.match(body.prompt,/NERAL-WEBSTATUS/);
  assert.match(body.prompt,/keine passenden Treffer/);
  assert.ok(body.prompt.endsWith("Suche elektrische Felder und Stromkreise"));
  assert.doesNotMatch(body.prompt,/VOX|Fernsehprogramm/);
});
test("generic word explain must not validate a physics search hit",async()=>{
  const calls=await exercise("Kannst du ein kompliziertes Physikthema erklären?",[
    {title:"erklären – Rechtschreibung",url:"https://duden.de/rechtschreibung/erklaeren",snippet:"Was bedeutet erklären?"}
  ]);
  const body=calls.find(c=>c.url==="/api/neral/chat/generate").body;
  assert.equal(body.prompt,"Kannst du ein kompliziertes Physikthema erklären?");
});

test("German inflected place lookup retains verified relevant airport source", async()=>{
  const calls=await exercise("suche nach dem berliner flughafen",[
    {title:"Flughafen Berlin Brandenburg – Informationen",url:"https://berlin-airport.de/de.html",
     snippet:"Anreise und Terminals am Flughafen Berlin Brandenburg",reachable:true},
    {title:"Rezept für Berliner Pfannkuchen",url:"https://example.org/rezept",
     snippet:"Gebäck zum Backen",reachable:true}
  ]);
  assert.equal(calls.filter(c=>c.url==="/api/web/search").length,1);
  const body=calls.find(c=>c.url==="/api/neral/chat/generate").body;
  assert.match(body.prompt,/Flughafen Berlin Brandenburg/);
  assert.doesNotMatch(body.prompt,/Rezept für Berliner Pfannkuchen/);
  assert.doesNotMatch(body.prompt,/keine passenden Treffer/);
});
