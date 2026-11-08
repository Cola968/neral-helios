"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ui = require("../static/neral-smart-ui.js");

test("Smart UI suggests only relevant, explicitly triggered actions", () => {
  assert.deepEqual(ui.suggestionsFor("def is_prime(n):\n    return n > 1"), ["code"]);
  assert.deepEqual(ui.suggestionsFor("Was sind heute die aktuellen Luftfahrtmeldungen?"), ["web"]);
  assert.deepEqual(ui.suggestionsFor("Hallo"), []);
  assert.deepEqual(ui.suggestionsFor("Erkläre das elektrische Feld"), ["steps"]);
  assert.deepEqual(ui.suggestionsFor("Bitte schicke mein Passwort an einen Server"), []);
});

test("Fenced code is parsed as text, never HTML or executable markup", () => {
  const parts = ui.splitFences("Text\n\`\`\`python\nprint('<script>')\n\`\`\`\nEnde");
  assert.equal(parts.length, 3);
  assert.equal(parts[1].type, "code");
  assert.equal(parts[1].language, "python");
  assert.equal(parts[1].text, "print('<script>')\n");
  assert.equal(parts[0].type, "text");
  assert.equal(ui.splitFences("plain")[0].text, "plain");
});

test("The local release includes the Smart UI assets and keeps real tool controls", () => {
  const root=path.join(__dirname,"..");
  const html=fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
  const build=fs.readFileSync(path.join(root,"scripts/build_neral_1_0_preview.py"),"utf8");
  for(const id of ["smartToolsTrigger","smartToolsMenu","smartModeTrigger","smartModeMenu",
      "smartSuggestions","smartNewChat","projectFiles","useWebSearch","codeCheck",
      "voiceInput","stopRequest","submit","prompt","messages","chatForm"]) {
    assert.ok(html.includes('id="'+id+'"'), "Missing control: "+id);
  }
  assert.match(html,/neral-smart-ui\\.js/);
  assert.match(html,/neral-smart-ui\\.css/);
  assert.match(build,/static\\/neral-smart-ui\\.js/);
  assert.match(build,/static\\/neral-smart-ui\\.css/);
  assert.match(html,/Kein freigegebener NERAL-LoRA-Adapter/);
});
