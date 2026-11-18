"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path");
const html=fs.readFileSync(path.join(__dirname,"../templates/neral_preview.html"),"utf8");
const js=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");

test("visible forced code-sandbox switch appears beside existing tier modes",()=>{
  assert.match(html, /id="forceSandbox"/);
  assert.match(html, /Code-Sandbox erzwingen/);
  assert.ok(html.indexOf('id="forceSandbox"')>html.indexOf('class="modes"'));
  assert.ok(html.indexOf('id="forceSandbox"')<html.indexOf('id="useWebSearch"'));
});

test("force flag reaches three-tier JSON transport, with no silent legacy fallback",()=>{
  assert.match(js,/force_sandbox:\s*forceSandbox/);
  assert.match(js,/selectedTier/);
  assert.match(js,/\/api\/neral\/chat\/three-tier/);
  assert.match(js,/forceSandbox\.checked/);
});

test("attachment multipart preserves force flag and selected tier",()=>{
  assert.match(js,/data\.append\("force_sandbox",/);
  assert.match(js,/data\.append\("tier",/);
  assert.match(js,/\/api\/neral\/preview\/attachment-chat/);
  assert.doesNotMatch(js,/Anhänge sind im experimentellen Drei-Modell-Modus noch nicht freigeschaltet/);
});
