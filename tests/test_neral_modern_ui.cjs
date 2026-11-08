"use strict";
// UI regression gate: modern chat shell must not remove security, voice or account controls.
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const root=path.resolve(__dirname,"..");
const read=p=>fs.readFileSync(path.join(root,p),"utf8");
const html=()=>read("templates/neral_preview.html");
const css=()=>read("static/neral-modern.css");
const js=()=>read("static/neral-modern.js");

test("responsive modern shell preserves every existing live tool and auth control",()=>{
  const h=html();
  for(const id of [
    "chatForm","prompt","submit","stopRequest","clear","useWebSearch",
    "projectFiles","voiceOutput","voiceSpeed","voiceAutoSend",
    "voiceInput","localVoiceStart","stopVoice","codeCheck",
    "attachmentChips","voiceStatus","localVoiceStatus","pluginStatus",
    "fileStatus","error","accountName","accountSettings",
    "newSavedChat","searchChats","chatSearchInput","savedChatList",
    "showLogin","logout","exportChats","accountDialog","accountForm",
    "accountUsername","accountPassword","accountRegister",
    "accountManageDialog","accountCurrentPassword","accountNewPassword",
    "connection","model","messages"
  ]){
    assert.equal((h.match(new RegExp('id="'+id+'"',"g"))||[]).length,1,id+" must exist exactly once");
  }
  for(const val of ['value="fast"','value="think"'])assert.ok(h.includes(val));
  assert.match(h,/neral-modern\.css/);
  assert.match(h,/neral-modern\.js/);
  assert.match(h,/id="sidebarToggle"/);
  assert.match(h,/id="sidebarBackdrop"/);
  assert.match(h,/id="extraTools"/);
  assert.match(h,/id="modelDetails"/);
  assert.match(h,/class="composer-box"/);
  assert.match(h,/type="file"[^>]*multiple|multiple[^>]*type="file"/);
});
test("compact layout uses full-height chat and keeps mobile nav accessible",()=>{
  const s=css();
  assert.match(s,/\.app-layout\s*\{[^}]*height:\s*100dvh/);
  assert.match(s,/\.messages\s*\{[^}]*overflow-y:\s*auto/);
  assert.match(s,/@media\s*\(max-width:\s*780px\)/);
  assert.match(s,/\.neral-sidebar\.is-open/);
  assert.match(s,/:focus-visible/);
  assert.match(s,/prefers-reduced-motion:\s*reduce/);
  assert.match(s,/#extraTools\[open\]/);
});
test("keyboard shortcuts and sidebar escape work without automatic submission",()=>{
  const s=js();
  assert.match(s,/event\.isComposing/);
  assert.match(s,/event\.shiftKey/);
  assert.match(s,/requestSubmit\(/);
  assert.match(s,/Escape/);
  assert.match(s,/aria-expanded/);
  assert.match(s,/dataset\.prompt/);
  assert.doesNotMatch(s,/innerHTML\s*=/);
});
test("local release includes the CSS and JS and offline checks are wired",()=>{
  const builder=read("scripts/build_neral_1_0_preview.py");
  const workflow=read(".github/workflows/neral-1-0-release-rc.yml");
  const nodeCheck=read(".github/workflows/neral-motion-ci.yml");
  for(const file of ["neral-modern.css","neral-modern.js"]){
    assert.ok(builder.includes('"static/'+file+'"'),file+" must ship in ZIP");
    assert.ok(workflow.includes("'static/"+file+"'"),file+" changes must rebuild ZIP");
  }
  assert.ok(nodeCheck.includes("tests/test_neral_modern_ui.cjs"));
  assert.ok(nodeCheck.includes("node --check static/neral-modern.js"));
});
