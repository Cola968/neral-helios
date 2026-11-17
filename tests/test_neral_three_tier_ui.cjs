"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
const root=path.join(__dirname,"..");
const source=fs.readFileSync(path.join(root,"static/neral-reference-ui.js"),"utf8");
const chat=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
const css=fs.readFileSync(path.join(root,"static/neral-reference-ui.css"),"utf8");

test("default preview keeps its existing two real LOW/HIGH stops",()=>{
  const html=fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
  assert.match(html,/id="modeEffortSlider"[^>]*max="1"/);
  assert.match(source,/let threeTier=false/);
  assert.match(source,/neral:three-tier-ready/);
});

test("three-tier only after server-confirmed event, middle stop routes to Terra",()=>{
  const listeners={};
  function el(id){
    const handlers={};
    return {id,hidden:false,value:"0",checked:false,attributes:{},dataset:{},
      textContent:"",children:[],classList:{toggle(){}},
      addEventListener(type,handler){(handlers[type]??=[]).push(handler)},
      dispatchEvent(event){for(const fn of handlers[event.type]||[])fn(event);return true},
      setAttribute(k,v){this.attributes[k]=v},
      append(...nodes){this.children.push(...nodes)},
      replaceChildren(...nodes){this.children=nodes},
      contains(target){return target===this},
      focus(){doc.activeElement=this},
      click(){this.dispatchEvent({type:"click"})},
      querySelector(selector){return selector===".mode-slider-ends"?endLabels:null}
    };
  }
  const names=["modeDropdownTrigger","modeDropdownMenu","modeOptionLow","modeOptionHigh",
    "modeDropdownLabel","modeEffortSlider","modeEffortTrack","modeSliderTitle","modeSliderReset"];
  const map=Object.fromEntries(names.map(name=>[name,el(name)]));
  const endLabels=el("endLabels");
  const group=el("modeGroup");
  const fast=el("fast"),think=el("think");
  fast.checked=true;
  const doc={
    activeElement:null,documentElement:{dataset:{}},
    getElementById:id=>map[id]||null,
    querySelector:selector=>selector===".modes"?group:
      selector.includes('value="fast"')?fast:
      selector.includes('value="think"')?think:null,
    createElement:tag=>el(tag),
    createTextNode:text=>({textContent:text}),
    addEventListener(t,fn){(listeners[t]??=[]).push(fn)}
  };
  const w={addEventListener:(t,fn)=>{(listeners[t]??=[]).push(fn)}};
  const Event=class{constructor(type){this.type=type}};
  vm.runInNewContext(source,{document:doc,window:w,Event,navigator:{},
    localStorage:{getItem:()=>null}},{timeout:1000});
  assert.equal(map.modeEffortSlider.max,undefined);
  assert.equal(map.modeDropdownLabel.textContent,"LOW");
  for(const fn of listeners["neral:three-tier-ready"]||[])fn();
  assert.equal(map.modeEffortSlider.max,"2");
  assert.equal(endLabels.children.map(x=>x.textContent).join(","),"Luna,Terra,Helios");
  map.modeEffortSlider.value="1";
  map.modeEffortSlider.dispatchEvent(new Event("input"));
  const middle=group.children[0].children[0];
  assert.equal(middle.value,"terra");
  assert.equal(middle.checked,true);
  assert.equal(map.modeDropdownLabel.textContent,"Terra");
  map.modeEffortSlider.value="2";
  map.modeEffortSlider.dispatchEvent(new Event("input"));
  assert.equal(think.checked,true);
  assert.equal(map.modeDropdownLabel.textContent,"Helios");
  map.modeSliderReset.click();
  assert.equal(fast.checked,true);
  assert.equal(map.modeDropdownLabel.textContent,"Luna");
});

test("HTTP chat changes only with pinned opt-in and never mislabels adapter output",()=>{
  assert.match(chat,/session\.three_tier_enabled===true/);
  assert.match(chat,/api\/neral\/chat\/three-tier\/status/);
  assert.match(chat,/path="\/api\/neral\/chat\/three-tier"/);
  assert.match(chat,/result\.runtime_tier_verified!==true/);
  assert.match(chat,/result\.adapter_loaded!==true/);
  assert.match(chat,/result\.model_generated===false/);
  assert.match(css,/data-level="terra"[^}]*width:50%/);
  assert.doesNotMatch(chat,/\.innerHTML\s*=/);
});
