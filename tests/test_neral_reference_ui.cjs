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
    "accountName","showLogin","messages","chatForm","prompt",
    "submit","stopRequest","voiceInput","localVoiceStart","extraTools","useWebSearch",
    "privacyDetails","sidebarToggle","chatSearchInput",
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
  assert.doesNotMatch(markup,/id="connection"/);
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


test("header omits connection and Qwen provenance chrome while local security remains enforced",()=>{
  const markup=html();
  const client=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const backend=fs.readFileSync(path.join(root,"neral_preview_server.py"),"utf8");
  assert.doesNotMatch(markup,/id="connection"|id="modelDetails"|id="model"/);
  assert.doesNotMatch(markup,/Modellherkunft|Lokales Qwen-Basismodell|Verbinde lokal/);
  assert.doesNotMatch(client,/\$\("connection"\)|\$\("model"\)/);
  assert.match(client,/status\.model_origin!==/);
  assert.match(client,/status\.adapter_loaded!==false/);
  assert.match(backend,/X-Neral-CSRF/);
  assert.match(markup,/id="error"/);
});


test("thinking-effort slider exposes only the two real NERAL routes",()=>{
  const markup=html(),sheet=css(),source=js();
  assert.match(markup,/id="modeEffortSlider"[^>]*type="range"[^>]*min="0"[^>]*max="1"[^>]*step="1"/);
  assert.match(markup,/id="modeSliderTitle"/);
  assert.match(markup,/id="modeSliderReset"/);
  assert.match(markup,/id="modeDropdownTrigger"[^>]*aria-haspopup="dialog"/);
  assert.match(markup,/id="modeDropdownMenu"[^>]*role="dialog"/);
  assert.match(sheet,/\.mode-effort-track/);
  assert.match(sheet,/::-webkit-slider-thumb/);
  assert.match(sheet,/::-moz-range-thumb/);
  assert.match(source,/modeEffortSlider/);
  assert.match(source,/addEventListener\("input"/);
  assert.match(source,/dispatchEvent\(new Event\("change"/);
});

test("slider and reset update the original FAST\/THINK radios without changing requests",()=>{
  const vm=require("node:vm");
  const listeners={};
  function element(id){
    const handlers={};
    return {
      id,hidden:false,value:"0",checked:false,textContent:"",dataset:{},attributes:{},
      classList:{toggle(){}},style:{},
      addEventListener(type,fn){(handlers[type]??=[]).push(fn)},
      dispatchEvent(event){for(const fn of handlers[event.type]||[])fn(event);return true},
      setAttribute(key,value){this.attributes[key]=value},
      contains(target){return target===this},
      focus(){doc.activeElement=this},click(){this.dispatchEvent({type:"click"})}
    };
  }
  const ids=["modeDropdownTrigger","modeDropdownMenu","modeOptionLow","modeOptionHigh",
    "modeDropdownLabel","modeEffortSlider","modeEffortTrack","modeSliderTitle","modeSliderReset","modeSliderCaption"];
  const map=Object.fromEntries(ids.map(id=>[id,element(id)]));
  map.modeDropdownMenu.hidden=true;
  const fast=element("fast"),think=element("think");
  fast.checked=true;
  const doc={
    activeElement:null,
    getElementById:id=>map[id]||null,
    querySelector:selector=>selector.includes('value="fast"')?fast:selector.includes('value="think"')?think:null,
    addEventListener(type,fn){(listeners[type]??=[]).push(fn)}
  };
  const E=class{constructor(type){this.type=type}};
  vm.runInNewContext(js(),{document:doc,Event:E,window:{},navigator:{},localStorage:{getItem:()=>null}},{timeout:1500});
  map.modeDropdownTrigger.click();
  assert.equal(map.modeDropdownMenu.hidden,false);
  assert.equal(doc.activeElement,map.modeEffortSlider);
  map.modeEffortSlider.value="1";
  map.modeEffortSlider.dispatchEvent(new E("input"));
  assert.equal(think.checked,true);
  assert.equal(fast.checked,false);
  assert.equal(map.modeDropdownLabel.textContent,"HIGH");
  assert.equal(map.modeSliderTitle.textContent,"Intensiv");
  assert.equal(map.modeDropdownMenu.hidden,false);
  map.modeSliderReset.click();
  assert.equal(fast.checked,true);
  assert.equal(think.checked,false);
  assert.equal(map.modeEffortSlider.value,"0");
  assert.equal(map.modeDropdownLabel.textContent,"LOW");
  map.modeDropdownMenu.dispatchEvent({type:"keydown",key:"Escape",preventDefault(){}});
  assert.equal(map.modeDropdownMenu.hidden,true);
});


test("Markdown bullets with bold labels render as real accessible list items, not literal asterisks",()=>{
  const vm=require("node:vm");
  const source=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const start=source.indexOf("  function renderReadableModelText(");
  const end=source.indexOf("  function append(",start);
  assert.ok(start>0&&end>start);
  class Element{
    constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.textContent="";}
    append(child){this.children.push(child)}
    get innerText(){return this.textContent+this.children.map(c=>c.innerText).join("")}
  }
  const renderer=vm.runInNewContext(source.slice(start,end)+"\nrenderReadableModelText",{
    document:{createElement:tag=>new Element(tag)}
  });
  const target=new Element("div");
  renderer(target,"Einleitung\n\n* **Bedeutung:** Ein Gruß\n* **Herkunft:** Ein altes Wort\n\nFazit.");
  const lists=target.children.filter(e=>e.tagName==="UL");
  assert.equal(lists.length,1);
  assert.equal(lists[0].children.length,2);
  assert.equal(lists[0].children[0].tagName,"LI");
  assert.equal(lists[0].children[0].children[0].tagName,"B");
  assert.equal(target.innerText.includes("* **"),false);
  const safe=new Element("div");
  renderer(safe,"* **Test:** <script>alert('x')</script>");
  assert.ok(safe.innerText.includes("<script>"));
  assert.equal(safe.children.some(e=>e.tagName==="SCRIPT"),false);
});

test("profile is a dedicated local dashboard, slider keeps two tactile stops",()=>{
 const markup=html(),sheet=css();
 for(const id of ["showProfile","profileDashboard","profileUsername","profileChatCount",
                  "profileMode","profileGuestActions","profileAccountActions","profileClose",
                  "profileOpenPassword","profileExportChats"]){
   assert.ok(markup.includes('id="'+id+'"'),"Missing profile element "+id);
 }
 assert.match(fs.readFileSync(path.join(root,"static/neral-accounts.css"),"utf8"),/\.profile-dashboard/);
 assert.match(markup,/mode-effort-stop/);
 assert.match(sheet,/\.mode-effort-stop/);
 assert.match(markup,/id="modeEffortSlider"[^>]*min="0"[^>]*max="1"/);
 assert.doesNotMatch(markup,/Premium freigeschaltet|Unbegrenzte Tokens/);
});


test("minimal NERAL slider contains only mode title, native range, end labels and reset",()=>{
  const markup=html();
  const section=markup.slice(markup.indexOf('id="modeDropdownMenu"'),markup.indexOf('id="modeDropdownMenu"')+2400);
  assert.ok(section.includes('id="modeSliderTitle"'));
  assert.ok(section.includes('id="modeSliderReset"'));
  assert.ok(section.includes('id="modeEffortSlider"'));
  assert.ok(section.includes('class="mode-slider-ends"'));
  assert.doesNotMatch(section,/mode-slider-bolt|id="modeSliderCaption"|NERAL · LOW|NERAL \/ CONTROL|role="status"/);
  assert.match(section,/aria-label="Denkmodus"/);
  assert.match(section,/min="0"[^>]*max="1"[^>]*step="1"/);
});

test("minimal mode picker preserves original two backend modes and separate profile",()=>{
  const markup=html(),source=js(),sheet=css();
  assert.match(markup,/id="profileDashboard"/);
  assert.match(markup,/name="mode" value="fast"/);
  assert.match(markup,/name="mode" value="think"/);
  assert.doesNotMatch(source,/modeSliderCaption/);
  assert.match(source,/sliderTitle\.textContent=titles\[selected\]/);
  assert.match(source,/sliderReset\.addEventListener\("click"/);
  assert.match(sheet,/\.mode-slider-copy/);
  assert.match(sheet,/\.mode-slider-ends/);
});
