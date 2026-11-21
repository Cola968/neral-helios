"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path");
const src=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");

test("code execution status is explicit and cannot be presented as verification",()=>{
  assert.match(src,/result\.tool_used === "helios_code_sandbox"/);
  assert.match(src,/result\.sandbox_executed === true/);
  assert.match(src,/result\.verified === false/);
  assert.match(src,/Werkzeug: Obsidian Code-Sandbox \(ausgeführt\)/);
  assert.match(src,/neral-tool-status/);
});
test("Luna and Terra may escalate only via declared unverified Helios origin",()=>{
  assert.match(src,/\["Luna","Terra"\]\.includes\(selectedTierLegacy\)/);
  assert.match(src,/result\.auto_escalated_from===selectedTierLegacy/);
  assert.match(src,/result\.safe_to_promote===false/);
});
