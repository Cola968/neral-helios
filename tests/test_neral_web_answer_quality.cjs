"use strict";
const test=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const path=require("node:path");
const src=fs.readFileSync(path.join(__dirname,"..","static","neral-preview.js"),"utf8");
const e2e=fs.readFileSync(path.join(__dirname,"..","scripts","neral_three_tier_browser_e2e.cjs"),"utf8");
const begin=src.indexOf("  function webEvidencePrompt(");
const end=src.indexOf("  function normaliseSearchRefusal(",begin);
assert.ok(begin>0&&end>begin);
const build=vm.runInNewContext(src.slice(begin,end)+"\nwebEvidencePrompt");
test("web prompt is short enough for bounded 96-token Luna and demands a topical answer",()=>{
  const hits=Array.from({length:3},(_,i)=>"[ "+i+" ] Berlin Brandenburg Flughafen Flugziele | https://berlin-airport.de | Informationen über Flugziele");
  const prompt=build("Suche im Internet nach aktuellen BER-Flugverbindungen.",hits,true,2);
  assert.ok(prompt.length<950,"The long meta-prompt + snippets displace the question and cause incomplete responses");
  assert.match(prompt,/höchstens 2 kurzen Sätzen|maximal 45 Wörter/i);
  assert.match(prompt,/unbestätig|ungeprüft/i);
  assert.match(prompt,/NERAL-WEBSTATUS/);
  assert.match(prompt,/URSPRÜNGLICHE FRAGE/);
});
test("web browser acceptance cannot pass with only links and a source-only placeholder",()=>{
  assert.match(e2e,/web_model_answer/);
  assert.match(e2e,/Websuche · Suchtreffer ohne Modellzusammenfassung|Suchtreffer ohne Modellzusammenfassung/);
  assert.match(e2e,/check\(.*web_model_answer|check\(.*generatedWebAnswer/s);
});
