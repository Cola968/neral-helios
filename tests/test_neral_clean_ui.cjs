"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const root=path.resolve(__dirname,"..");
const read=file=>fs.readFileSync(path.join(root,file),"utf8");
const html=()=>read("templates/neral_preview.html");
const ids=text=>[...text.matchAll(/id="([^"]+)"/g)].map(x=>x[1]);
test("quiet composer has one plus menu for uploads/formulas and still preserves controls",()=>{
  const h=html();
  const composer=h.slice(h.indexOf('<section class="composer-panel"'),h.indexOf('<section class="neral-formula-library"'));
  assert.match(composer,/<details id="quickAddMenu"[^>]*>/);
  const menu=composer.slice(composer.indexOf('id="quickAddMenu"'),composer.indexOf('id="quickAddMenu"')+2200);
  assert.match(menu,/id="projectFiles"/);
  assert.match(menu,/id="formulaLibraryToggle"/);
  assert.match(menu,/aria-label="Weitere Inhalte hinzufügen"/);
  for(const id of ["useWebSearch","modeDropdownTrigger","localVoiceStart","stopRequest","submit"])
    assert.ok(composer.includes(`id="${id}"`),id+" needs to remain directly accessible");
  const quickActions=composer.slice(composer.indexOf('class="actions composer-actions"'),composer.indexOf('id="stopRequest"'));
  assert.ok(!quickActions.includes('id="voiceInput"'),"Browser dictation moves to advanced tools, not removed");
});
test("advanced menu owns browser voice, theme and protected existing tools",()=>{
  const h=html(),panel=h.slice(h.indexOf('class="tools-panel"'),h.indexOf('class="tools-panel"')+4300);
  for(const id of ["voiceInput","themeToggle","codeCheck","clear","showPrivacy","stopVoice","voiceOutput","voiceAutoSend","voiceSpeed"])
    assert.match(panel,new RegExp(`id="${id}"`),id+" stays reachable");
  const all=ids(h);
  for(const id of ["projectFiles","voiceInput","themeToggle","formulaLibraryToggle","extraTools","modeDropdownTrigger","localVoiceStart","modelDetails"])
    assert.equal(all.filter(x=>x===id).length,1,id+" has exactly one live control");
});
test("design has calm centered message column and dark, mobile and keyboard focus states",()=>{
  const s=read("static/neral-clean-ui.css");
  for(const token of [".composer-panel",".composer-box",".quick-add-panel",".message.assistant",".neral-sidebar",".masthead"])
    assert.ok(s.includes(token),token);
  assert.match(s,/@media\s*\(max-width:\s*520px\)/);
  assert.match(s,/html\[data-theme=dark\]/);
  assert.match(s,/:focus-visible/);
  assert.match(s,/prefers-reduced-motion/);
  assert.doesNotMatch(s,/background:\s*linear-gradient/);
});
test("quick menu closes on action and escape, leaves file/voice business logic untouched",()=>{
  const js=read("static/neral-clean-ui.js");
  assert.doesNotMatch(js,/\binnerHTML\s*=|fetch\(|\beval\s*\(/);
  const listeners=new Map();
  const make=id=>({id,open:false,contains:t=>t?.id===id,events:new Map(),
    addEventListener(n,fn){this.events.set(n,fn)},fire(n){this.events.get(n)?.({target:this})},focus(){}});
  const quick=make("quickAddMenu"),formulas=make("formulaLibraryToggle"),input=make("projectFiles"),model=make("modelDetails"),tools=make("extraTools");
  const map={quickAddMenu:quick,formulaLibraryToggle:formulas,projectFiles:input,modelDetails:model,extraTools:tools};
  const document={getElementById:id=>map[id]||null,addEventListener:(n,fn)=>listeners.set(n,fn)};
  vm.runInNewContext(js,{document,window:{},console},{timeout:1000});
  quick.open=true;formulas.fire("click");assert.equal(quick.open,false);
  quick.open=true;input.fire("change");assert.equal(quick.open,false);
  quick.open=true;listeners.get("keydown")({key:"Escape",preventDefault(){}});assert.equal(quick.open,false);
  quick.open=true;listeners.get("pointerdown")({target:{id:"outside"}});assert.equal(quick.open,false);
});
test("ZIP ships new UI and both CI workflows validate it",()=>{
  const h=html(),bundle=read("scripts/build_neral_1_0_preview.py"),
    motion=read(".github/workflows/neral-motion-ci.yml"),
    release=read(".github/workflows/neral-1-0-release-rc.yml");
  for(const file of ["neral-clean-ui.js","neral-clean-ui.css"]){
    assert.match(h,new RegExp(`/static/${file.replaceAll(".","\\.")}`));
    assert.ok(bundle.includes(`"static/${file}"`));
    assert.ok(release.includes(`'static/${file}'`));
  }
  assert.ok(motion.includes("tests/test_neral_clean_ui.cjs"));
  assert.ok(motion.includes("node --check static/neral-clean-ui.js"));
});
