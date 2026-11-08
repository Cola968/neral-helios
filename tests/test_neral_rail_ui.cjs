"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path");
const ui=require("../static/neral-rail-ui.js");

test("train route is detected in conversational German without requiring a special command",()=>{
  assert.deepEqual(ui.parse("Zeig mir eine passende Verbindung von Berlin Hauptbahnhof nach Stuttgart"),{
    from:"Berlin Hauptbahnhof",to:"Stuttgart"
  });
  assert.deepEqual(ui.parse("Suche eine Zugverbindung von Berlin Hbf nach Stuttgart Hbf morgen um 09:30"),{
    from:"Berlin Hbf",to:"Stuttgart Hbf",when:"morgen",time:"09:30"
  });
  assert.deepEqual(ui.parse("Zug von Hamburg Hbf nach Köln Hbf ab 8 Uhr"),{
    from:"Hamburg Hbf",to:"Köln Hbf",time:"08:00"
  });
  for(const question of [
    "Wie funktioniert eine SQL-Verbindung?",
    "Führe die Variablen von Berlin nach Stuttgart",
    "Hallo",
    "Zug von https://evil.test nach Berlin",
    "Zeig mir eine Verbindung von X nach Y am 14.12.2027",
  ]) assert.equal(ui.parse(question),null,question);
});

test("real geometry is split at missing sections rather than invented as air-line",()=>{
  const segments=ui.geometrySegments([[52.525,13.369],[52.52,13.362],[50.2,11],[48.78,9.18],[48.779,9.178]]);
  assert.equal(segments.length,2);
  assert.equal(segments[0].length,2);
  assert.equal(segments[1].length,2);
  assert.deepEqual(ui.geometrySegments([]),[]);
  assert.deepEqual(ui.geometrySegments([[52,13],[48,9]]),[]);
});

test("provider data must validate before a journey is presented",()=>{
  const valid={
    status:"ok",from:"Berlin Hbf",to:"Stuttgart Hbf",
    source:"BahnConnections / Transitous",updatedAt:"2026-11-09T09:00:00+01:00",
    journeys:[{
      startTime:"2026-11-09T09:10:00+01:00",endTime:"2026-11-09T14:20:00+01:00",
      durationSeconds:18600,transfers:0,cancelled:false,realtimeStatus:"partial",
      legs:[{name:"ICE 999",category:"fern",from:{name:"Berlin Hbf",lat:52.52,lon:13.37},
        to:{name:"Stuttgart Hbf",lat:48.78,lon:9.18},
        startTime:"2026-11-09T09:10:00+01:00",endTime:"2026-11-09T14:20:00+01:00",
        points:[[52.52,13.37],[52.519,13.368]]}]
    }]
  };
  assert.equal(ui.validate(valid).journeys.length,1);
  assert.equal(ui.validate({...valid,journeys:[{...valid.journeys[0],startTime:"not-time"}]}),null);
  assert.equal(ui.validate({...valid,journeys:[{...valid.journeys[0],legs:[{...valid.journeys[0].legs[0],name:"<script>alert(1)</script>"}]}]}).journeys.length,1);
  assert.equal(ui.validate({...valid,journeys:[{...valid.journeys[0],legs:[]}]}),null);
});

test("release preview contains local BahnConnections bridge and frontend scripts",()=>{
  const root=path.join(__dirname,"..");
  const html=fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
  const build=fs.readFileSync(path.join(root,"scripts/build_neral_1_0_preview.py"),"utf8");
  const preview=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const source=fs.readFileSync(path.join(root,"static/neral-rail-ui.js"),"utf8");
  for(const asset of ["neral-rail-ui.js","neral-rail-ui.css"]) {
    assert.ok(html.includes(asset),asset);
    assert.ok(build.includes(asset),asset);
  }
  assert.ok(build.includes("neral_bahnconnections.py"));
  assert.ok(preview.includes("/api/neral/preview/journeys"));
  assert.ok(preview.includes("NERALJourneyUI"));
  assert.doesNotMatch(source,/\b(?:eval|new Function)\s*\(/);
  assert.doesNotMatch(source,/\.innerHTML\s*=/);
});
