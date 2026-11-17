"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
const source=fs.readFileSync(path.join(__dirname,"..","static","neral-preview.js"),"utf8");
const start=source.indexOf("  function sourceOnlyWebFallback(");
const end=source.indexOf("  function appendSources(",start);
assert.ok(start>0&&end>start,"Source-only fallback predicate must be present");
const {sourceOnlyWebFallback}=vm.runInNewContext(
  "(function(){"+source.slice(start,end)+"\nreturn {sourceOnlyWebFallback};})()");
const modelFailure=new Error("Drei-Modell-Laufzeit nicht bereit oder Modellantwort unvollständig.");
const discovered=[{url:"https://ber.berlin-airport.de/de.html",discovered:true,topically_relevant:true,reachable:false}];

test("verified provider discovery is displayed only after model runtime 503",()=>{
  assert.equal(sourceOnlyWebFallback(modelFailure,true,discovered,true),true);
});
test("never pretend a source-only reply is a normal AI answer or bypass unrelated failures",()=>{
  assert.equal(sourceOnlyWebFallback(new Error("Lokale NERAL-Sitzung fehlt."),true,discovered,true),false);
  assert.equal(sourceOnlyWebFallback(modelFailure,false,discovered,true),false);
  assert.equal(sourceOnlyWebFallback(modelFailure,true,discovered,false),false);
  assert.equal(sourceOnlyWebFallback(modelFailure,true,[],true),false);
  assert.equal(sourceOnlyWebFallback(modelFailure,true,[{url:"https://example.org",discovered:true,topically_relevant:false}],true),false);
  assert.equal(sourceOnlyWebFallback(modelFailure,true,[{url:"http://localhost:9690",reachable:true}],true),false);
});
