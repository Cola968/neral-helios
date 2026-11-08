"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const {createPainter}=require("../static/neral-live-stream.js");
function frames(){
  const jobs=new Map();let next=0;
  return {
    frame(cb){const id=++next;jobs.set(id,cb);return id;},
    cancel(id){jobs.delete(id);},
    run(){const work=[...jobs.values()];jobs.clear();for(const fn of work)fn();},
    pending(){return jobs.size},
    async drain(max=300){
      for(let i=0;i<max && jobs.size;i++){this.run();await Promise.resolve();}
      assert.equal(jobs.size,0,"Painter never drains");
    }
  };
}
test("rendering a complete HIGH response displays text progressively rather than all at once",async()=>{
  const f=frames(),target={textContent:""},p=createPainter(target,{schedule:f.frame,cancelFrame:f.cancel});
  p.push("Hallo. Hier ist die Antwort mit einigen Details und 1234567890.");
  assert.equal(target.textContent,"");
  f.run();
  assert.ok(target.textContent.length>0);
  assert.ok(target.textContent.length<60,"one frame must not dump whole response");
  const done=p.finish();
  await f.drain();
  assert.equal(await done,true);
  assert.equal(target.textContent,"Hallo. Hier ist die Antwort mit einigen Details und 1234567890.");
});
test("large Ollama chunk is painted in steps, and later chunks are appended in correct order",async()=>{
  const f=frames(),target={textContent:""},p=createPainter(target,{schedule:f.frame,cancelFrame:f.cancel});
  p.push("Abschnitt A ".repeat(40));
  f.run();
  const part=target.textContent;
  assert.ok(part.length>0&&part.length<400);
  p.push("🛤️ Abschnitt B");
  const finished=p.finish();
  await f.drain();
  assert.equal(await finished,true);
  assert.equal(target.textContent,"Abschnitt A ".repeat(40)+"🛤️ Abschnitt B");
});
test("abort cancels frame updates and never reveals discarded private chunks",async()=>{
  const f=frames(),target={textContent:""},active={value:true};
  const p=createPainter(target,{schedule:f.frame,cancelFrame:f.cancel,isCurrent:()=>active.value});
  p.push("HALBGEHEIM");
  f.run();active.value=false;
  const old=target.textContent;
  assert.equal(await p.finish(),false);
  f.run();assert.equal(target.textContent,old);
});
test("reduced motion and hidden document show full text instantly",async()=>{
  for(const opts of [{reducedMotion:true},{isHidden:()=>true}]){
    const f=frames(),target={textContent:""},p=createPainter(target,{...opts,schedule:f.frame,cancelFrame:f.cancel});
    p.push("Vollständig");
    assert.equal(await p.finish(),true);
    assert.equal(target.textContent,"Vollständig");
  }
});
test("late text after completion and cancel cannot mutate another conversation",async()=>{
  const f=frames(),target={textContent:""},p=createPainter(target,{schedule:f.frame,cancelFrame:f.cancel});
  p.push("A");
  const completed=p.finish();await f.drain();
  assert.equal(await completed,true);
  p.push("NEIN");f.run();assert.equal(target.textContent,"A");
  p.cancel();p.push("NEIN");f.run();assert.equal(target.textContent,"A");
});
test("the live client uses the same painter for LOW network deltas and HIGH verified final answers",()=>{
  const fs=require("node:fs"),path=require("node:path");
  const src=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
  assert.match(src,/NeralLiveStream\?\.createPainter/);
  assert.match(src,/painter\.push\(piece\)/);
  assert.match(src,/painter\.push\(result\.text\)/);
  assert.match(src,/await painter\.finish\(\)/);
  assert.match(src,/painter\?\.cancel\(\)/);
  assert.match(src,/mode==="think"/);
});
