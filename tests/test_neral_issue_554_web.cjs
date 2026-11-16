"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const vm=require("node:vm"),fs=require("node:fs"),path=require("node:path");
const src=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
function extract(start,end,expose){
  const i=src.indexOf(start),j=src.indexOf(end,i);
  assert.ok(i>0&&j>i,`Missing ${start}`);
  return vm.runInNewContext(src.slice(i,j)+"\n"+expose);
}
const web=extract("  function skipWebForLocalBasics(","  // Reuse only a previous query", "({shouldSearchWeb})");
const status=extract("  function webEvidencePrompt(","  function relevantSearchResult(", "({webEvidencePrompt,normaliseSearchRefusal})");

test("Web opt-in finds public current BER airline route question without explicit keyword",()=>{
  assert.equal(web.shouldSearchWeb("Welche Airlines fliegen vom BER nach Kanada?"),true);
  assert.equal(web.shouldSearchWeb("Welche Fluggesellschaften fliegen ab Berlin nach Toronto?"),true);
  assert.equal(web.shouldSearchWeb("Wer ist derzeit CEO von Lufthansa?"),true);
});
test("do not search normal chat, private information or small talk",()=>{
  for(const q of ["Hallo NERAL!","Wer bist du?","Wie geht es dir?",
      "Ich bin heute sehr traurig, können wir reden?","Was soll ich meiner Freundin sagen?",
      "Wie funktioniert ein Kondensator?"]){
    assert.equal(web.shouldSearchWeb(q),false,q);
  }
});
test("WEBSTATUS accurately distinguishes reachable checks and previews",()=>{
  const prompt=status.webEvidencePrompt("BER",["[1] Treffer A","[2] Treffer B","[3] Treffer C"],true,1);
  assert.match(prompt,/1 (?:erreichbar|geprüft)/i);
  assert.match(prompt,/2 (?:unbestätigt|ungeprüft|nicht geprüft)/i);
  assert.doesNotMatch(prompt,/3 erreichbare/i);
});
test("WEBSTATUS refusal text does not claim mixed results all reachable",()=>{
  const fix=status.normaliseSearchRefusal("Ich kann keine externe Websuche durchführen.",3,true,1);
  assert.doesNotMatch(fix,/3 erreichbare/);
  assert.match(fix,/1 (?:erreichbar|geprüft)/i);
});
