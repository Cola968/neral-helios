"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const root=path.join(__dirname,"..");
const html=()=>fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
const css=()=>fs.readFileSync(path.join(root,"static/neral-reference-ui.css"),"utf8");
const js=()=>fs.readFileSync(path.join(root,"static/neral-reference-ui.js"),"utf8");

test("screenshot-based UI ships with the complete local release",()=>{
  const markup=html(),build=fs.readFileSync(path.join(root,"scripts/build_neral_1_0_preview.py"),"utf8");
  for(const name of ["neral-reference-ui.css","neral-reference-ui.js"]){
    assert.match(markup,new RegExp(name.replaceAll(".","\\.")));
    assert.ok(build.includes("static/"+name),"Not included in release: "+name);
  }
});
test("real chat, account, attachments, voice and model controls survive redesign",()=>{
  const markup=html();
  for(const id of [
    "neralSidebar","savedChatList","newSavedChat","searchChats","projectFiles",
    "accountName","showLogin","model","connection","messages","chatForm","prompt",
    "submit","stopRequest","voiceInput","localVoiceStart","extraTools","useWebSearch",
    "privacyDetails","modelDetails","sidebarToggle","chatSearchInput",
    "modeDropdownTrigger","modeDropdownMenu","modeOptionLow","modeOptionHigh",
    "themeToggle","projectFilesShortcut"
  ])assert.ok(markup.includes('id="'+id+'"'),"Missing "+id);
  assert.match(markup,/name="mode" value="fast"/);
  assert.match(markup,/name="mode" value="think"/);
  assert.doesNotMatch(markup,/Python-Code überprüfen<\/button>|Elektrisches Feld verstehen<\/button>/);
});
test("no pseudo conversations, flashy hero prose, or fake connection claims",()=>{
  const markup=html();
  assert.doesNotMatch(markup,/Code debuggen<\/button>|Frage analysieren<\/button>/);
  assert.doesNotMatch(markup,/Dein intelligenter Begleiter|Willkommen in der Zukunft|Erlebe die Zukunft/);
  assert.match(markup,/id="connection"/);
  assert.match(markup,/id="savedChatList"/);
  assert.ok(!markup.includes('id="fakeHistory"'));
});
test("dropdown always selects existing FAST/THINK radio, never changes the chat request path",()=>{
  const source=js();
  assert.match(source,/querySelector\('input\[name="mode"\]\[value="fast"\]'\)/);
  assert.match(source,/querySelector\('input\[name="mode"\]\[value="think"\]'\)/);
  assert.match(source,/aria-expanded/);
  assert.match(source,/Escape/);
  assert.match(source,/pointerdown/);
  assert.doesNotMatch(source,/\b(?:eval|Function)\s*\(/);
  assert.doesNotMatch(source,/\.innerHTML\s*=/);
});
test("custom CSS uses the reference proportions and has real mobile layout",()=>{
  const sheet=css();
  assert.match(sheet,/\.neral-sidebar/);
  assert.match(sheet,/\.messages/);
  assert.match(sheet,/\.composer-panel/);
  assert.match(sheet,/\.mode-dropdown/);
  assert.match(sheet,/@media\(max-width:\s*780px\)/);
  assert.match(sheet,/prefers-reduced-motion/);
  assert.doesNotMatch(sheet,/linear-gradient\(|backdrop-filter|glow|neon|glassmorphism/i);
});
