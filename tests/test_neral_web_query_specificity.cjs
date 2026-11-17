"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");

const src=fs.readFileSync(path.join(__dirname,"..","static","neral-preview.js"),"utf8");
const begin=src.indexOf("  function isSearchFollowup(");
const end=src.indexOf("  // Snippets are untrusted evidence",begin);
assert.ok(begin>=0 && end>begin, "Query preparation functions must remain present");
const {makeSearchQuery}=vm.runInNewContext(
  "(function(){"+src.slice(begin,end)+"
return {makeSearchQuery};})()");

test("explicit BER flight-connection search disambiguates the airport and drops vague freshness filler",()=>{
  assert.equal(makeSearchQuery("Suche im Internet nach aktuellen BER-Flugverbindungen."),
    "BER Berlin Brandenburg Flughafen Flugziele");
});
test("unrelated topical and fresh news searches are not rewritten",()=>{
  assert.equal(makeSearchQuery("Suche im Internet nach aktuellen Nachrichten zur KI."),
    "Nachrichten zur KI.");
  assert.equal(makeSearchQuery("Suche im Internet nach T4 Grafikkarte"),
    "T4 Grafikkarte");
});
test("EZB query removes conversational framing and preserves freshness",()=>{
  assert.equal(makeSearchQuery("Suche bitte aktuell im Web: Wie hoch ist heute der Leitzins der EZB? Nenne mir deine Quellen."),
    "Leitzins EZB aktuell");
});
test("empty web search subjects and followups preserve their original contracts",()=>{
  assert.equal(makeSearchQuery("Suche im Internet nach"),"");
  assert.equal(makeSearchQuery("Warum?","Wetter Berlin"),"Wetter Berlin Warum?");
});
