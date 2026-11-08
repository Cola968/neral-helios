"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const root=path.resolve(__dirname,"..");
const fence=String.fromCharCode(96).repeat(3);
function boot(){
  const window={};
  class El{
    constructor(tag="div"){this.tagName=tag.toUpperCase();this.children=[];this.attrs={};this.value="";this.textContent="";this.className="";this.events=new Map();this.dataset={};}
    append(...nodes){this.children.push(...nodes);}
    replaceChildren(...nodes){this.children=[...nodes];}
    setAttribute(k,v){this.attrs[k]=String(v);}
    getAttribute(k){return this.attrs[k]??null;}
    addEventListener(k,fn){this.events.set(k,fn);}
    fire(k){this.events.get(k)?.({target:this,preventDefault(){}});}
  }
  const document={createElement:tag=>new El(tag)};
  const src=fs.readFileSync(path.join(root,"static/neral-dynamic-ui.js"),"utf8");
  vm.runInNewContext(src,{window,document,Intl,Math,Number,console},{filename:"dynamic-ui.js",timeout:1500});
  return {...window.NeralDynamicUI,document,El};
}
const flatten=node=>[String(node.textContent||""),...node.children.map(flatten)].join(" ");
test("regular prose and unsupported live maps remain plain text",()=>{
  const ui=boot();
  assert.equal(ui.plan({prompt:"Hallo",text:"Guten Tag",result:{model_generated:true}}),null);
  assert.equal(ui.plan({prompt:"Fahrplan Berlin Stuttgart",text:"Keine Live-Daten vorhanden",result:{model_generated:true}}),null);
});
test("data table becomes sortable and filterable without inventing rows",()=>{
  const ui=boot();
  const spec=ui.plan({prompt:"Vergleiche die Werte",text:"| Stadt | Fahrgäste |\n|---|---:|\n| Berlin | 25 |\n| Hamburg | 10 |",result:{model_generated:true}});
  assert.equal(spec.kind,"table");
  assert.deepEqual([...spec.headers],["Stadt","Fahrgäste"]);
  assert.deepEqual([...spec.rows[0]],["Berlin","25"]);
  assert.equal(spec.provenance,"model_unverified");
  const target=new ui.El();ui.mount(target,spec);
  assert.match(flatten(target),/Berlin/);
});
test("explicit chart request uses only model-supplied numeric rows",()=>{
  const ui=boot();
  const spec=ui.plan({prompt:"Zeige ein Balkendiagramm",text:"| Jahr | Reisende |\n|---|---:|\n| 2024 | 12,5 |\n| 2025 | 20,0 |",result:{model_generated:true}});
  assert.equal(spec.kind,"bar");
  assert.deepEqual([...spec.points].map(x=>x.value),[12.5,20]);
  assert.equal(spec.provenance,"model_unverified");
  const target=new ui.El();ui.mount(target,spec);
  assert.match(flatten(target),/2025/);
});
test("verified arithmetic has a functioning calculator with safe arithmetic",()=>{
  const ui=boot();
  const spec=ui.plan({prompt:"Berechne 12 * 3",text:"36",result:{model_generated:false,tool_used:"neral_math_exact",exact:true}});
  assert.equal(spec.kind,"calculator");
  assert.equal(spec.provenance,"verified_local");
  assert.equal(ui.calculate(12,3,"*"),36);
  assert.equal(ui.calculate(12,0,"/"),null);
  const target=new ui.El();ui.mount(target,spec);
  assert.match(flatten(target),/36/);
});
test("comparison bullets become selectable cards",()=>{
  const ui=boot();
  const spec=ui.plan({prompt:"Vergleiche LOW und HIGH",text:"- **LOW:** Schneller Alltagschat.\n- **HIGH:** Prüfung komplexer Fragen.",result:{model_generated:true}});
  assert.equal(spec.kind,"cards");assert.equal(spec.items.length,2);
  const target=new ui.El();ui.mount(target,spec);
  assert.match(flatten(target),/LOW/);assert.match(flatten(target),/HIGH/);
});
test("structured widgets are allowlisted, DOM safe and cannot load URLs",()=>{
  const ui=boot();
  const reply="Vergleich\n"+fence+"neral-ui\n"+JSON.stringify({version:1,type:"cards",title:"Optionen",items:[{title:"A",detail:"<img src=x onerror=alert(1)>"},{title:"B",detail:"Text"}]})+"\n"+fence;
  const plan=ui.plan({prompt:"Optionen",text:reply,result:{model_generated:true}});
  assert.equal(plan.kind,"cards");
  const target=new ui.El();ui.mount(target,plan);
  assert.match(flatten(target),/<img src=x onerror=alert\(1\)>/);
  for(const bad of [{version:1,type:"iframe",src:"https://evil.example"},{version:1,type:"cards",items:[{title:"A",detail:"B",url:"https://evil"}]},{version:1,type:"cards",items:[{title:"A",detail:"B"}],html:"<script>1</script>"}]){
    const raw=fence+"neral-ui\n"+JSON.stringify(bad)+"\n"+fence;
    assert.equal(ui.plan({prompt:"Optionen",text:raw,result:{model_generated:true}}),null);
  }
  const source=fs.readFileSync(path.join(root,"static/neral-dynamic-ui.js"),"utf8");
  assert.doesNotMatch(source,/innerHTML|outerHTML|document\.write|new Function|\beval\s*\(/);
});
test("oversized and nonnumeric data fails closed",()=>{
  const ui=boot();
  assert.equal(ui.plan({prompt:"Tabelle",text:"X".repeat(12000),result:{}}),null);
  assert.equal(ui.plan({prompt:"Diagramm",text:"| Ort | Wert |\n|---|---|\n| A | NaN |\n| B | Infinity |",result:{}}),null);
});
test("frontend and installed ZIP actually include the renderer",()=>{
  const preview=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const html=fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
  const builder=fs.readFileSync(path.join(root,"scripts/build_neral_1_0_preview.py"),"utf8");
  const workflow=fs.readFileSync(path.join(root,".github/workflows/neral-motion-ci.yml"),"utf8");
  assert.match(preview,/NeralDynamicUI/);
  assert.match(preview,/ui\.mount/);
  assert.match(html,/\/static\/neral-dynamic-ui\.css/);
  assert.match(html,/\/static\/neral-dynamic-ui\.js/);
  assert.ok(builder.includes('"static/neral-dynamic-ui.css"'));
  assert.ok(builder.includes('"static/neral-dynamic-ui.js"'));
  assert.ok(workflow.includes("tests/test_neral_dynamic_ui.cjs"));
});
