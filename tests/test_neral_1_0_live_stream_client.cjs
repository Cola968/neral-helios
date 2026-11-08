"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path");
const api=require("../static/neral-live-stream.js");
const root=path.join(__dirname,"..");
const encoder=new TextEncoder();
function response(chunks,ok=true){
  return {ok,headers:{get:()=>ok?"application/x-ndjson":"application/json"},
    body:{getReader(){
      let i=0;
      return {async read(){return i<chunks.length?
        {done:false,value:encoder.encode(chunks[i++])}:{done:true,value:undefined};},
        releaseLock(){},cancel(){}};
    }},
    async json(){return {error:"NERAL beschäftigt"};},
  };
}
const ev=(type,data)=>JSON.stringify({type,...data})+"\n";
test("NDJSON decoder emits actual chunks before final and preserves UTF-8",async()=>{
  const events=[ev("delta",{text:"Guten "}),ev("delta",{text:"Morgen 🌍"}),
    ev("final",{result:{text:"Guten Morgen 🌍",model_origin:"local_foundation_not_neral_adapter",adapter_loaded:false}})];
  const joined=events.join("");
  const chunks=[joined.slice(0,3),joined.slice(3,46),joined.slice(46,joined.length-9),joined.slice(-9)];
  const received=[];
  const result=await api.read(response(chunks),{onDelta:s=>received.push(s)});
  assert.deepEqual(received,["Guten ","Morgen 🌍"]);
  assert.equal(result.text,"Guten Morgen 🌍");
});
test("tool-only final is valid without fabricated tokens",async()=>{
  const x=await api.read(response([ev("final",{result:{text:"4",model_generated:false}})]));
  assert.equal(x.text,"4");
});
test("never accept incomplete, contradictory, or oversized streams",async()=>{
  const bad=[
    [ev("delta",{text:"PARTIAL"})],
    [ev("delta",{text:"A"}),ev("final",{result:{text:"B"}})],
    [ev("delta",{text:"A"}),ev("error",{error:"broken"})],
    [ev("final",{result:{text:"Done"}}),ev("delta",{text:"EXTRA"})],
    [ev("delta",{text:"x".repeat(34000)})],
    ["{invalid}\n"],
  ];
  for(const chunks of bad){
    await assert.rejects(()=>api.read(response(chunks)),Error);
  }
});
test("server-side 409 is surfaced as a safe message",async()=>{
  await assert.rejects(()=>api.read(response([],false)),/beschäftigt/);
});
test("conversation script displays user request before network, uses true stream only in LOW",()=>{
  const src=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const a=src.indexOf('async function sendMessage('),b=src.indexOf('form.addEventListener("submit"');
  const send=src.slice(a,b);
  assert.ok(send.indexOf('append("user",')>=0);
  assert.ok(send.indexOf('append("user",')<send.indexOf('await json('));
  assert.match(send,/\/api\/neral\/chat\/generate-stream/);
  assert.match(send,/NeralLiveStream\.read/);
  assert.match(send,/mode==="generate"/);
  assert.match(send,/startedEpoch!==conversationEpoch/);
});
test("release includes browser client and route and no simulated typing timer",()=>{
  const html=fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
  const build=fs.readFileSync(path.join(root,"scripts/build_neral_1_0_preview.py"),"utf8");
  const workflow=fs.readFileSync(path.join(root,".github/workflows/neral-1-0-release-rc.yml"),"utf8");
  assert.match(html,/neral-live-stream\.js/);
  assert.match(build,/static\/neral-live-stream\.js/);
  assert.match(workflow,/test_neral_1_0_live_streaming/);
  assert.doesNotMatch(fs.readFileSync(path.join(root,"static/neral-live-stream.js"),"utf8"),/setInterval|setTimeout|innerHTML|eval\(/);
});
