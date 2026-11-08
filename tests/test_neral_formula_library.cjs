"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const base=path.resolve(__dirname,".."),read=p=>fs.readFileSync(path.join(base,p),"utf8");

function load(dom=null){
  const win={NeralMath:{render(){return true;}}};
  const document=dom||{getElementById:()=>null};
  const ctx={window:win,document,console,Event:class{constructor(type){this.type=type;}}};
  vm.runInNewContext(read("static/neral-formulas.js"),ctx,{timeout:2500});
  return win.NeralFormulas;
}
test("searchable mathematical and physical formula catalog is built into the app",()=>{
  const api=load();
  assert.ok(api,"formula catalog needs public read-only API");
  assert.ok(api.catalog.length>=42);
  assert.equal(new Set(api.catalog.map(x=>x.id)).size,api.catalog.length);
  assert.ok(api.catalog.every(x=>
    /^[a-z0-9-]+$/.test(x.id)&&
    typeof x.name==="string"&&x.name.length<100&&
    typeof x.latex==="string"&&x.latex.length<400&&
    typeof x.category==="string"&&typeof x.note==="string"));
  for(const section of ["Algebra","Analysis","Mechanik","Relativität","Elektromagnetismus","Astronomie","Chemie","Statistik"])
    assert.ok(api.catalog.some(x=>x.category===section),section+" missing");
  assert.ok(api.search("schwarzschild").some(x=>x.id==="schwarzschild"));
  assert.ok(api.search("kepler").length>0);
  assert.ok(api.search("rechnerisch").length===0);
  assert.ok(api.search("energie","Relativität").length>0);
});
test("catalog formulas genuinely render with pinned locally bundled KaTeX and mhchem",()=>{
  const katex=require(path.join(base,"static/vendor/katex/katex.min.js"));
  require(path.join(base,"static/vendor/katex/mhchem.min.js"));
  for(const item of load().catalog){
    const html=katex.renderToString(item.latex,{
      throwOnError:true,trust:false,strict:"warn",output:"htmlAndMathml"
    });
    assert.match(html,/<math[\s>]/,item.id);
    assert.match(html,/class="katex"/,item.id);
  }
});
test("formula picker inserts at caret without sending, deleting draft or exceeding 4000 characters",()=>{
  const api=load();
  const snippet=api.insertText("Erkläre: Ende",9,9,"$E=mc^2$",4000);
  assert.equal(snippet.text,"Erkläre: $E=mc^2$Ende");
  assert.ok(snippet.caret>9);
  assert.equal(api.insertText("x".repeat(3999),3999,3999,"$E=mc^2$",4000),null);
  assert.equal(api.insertText("<script>",0,0,"$E=mc^2$",4000).text,
               "$E=mc^2$<script>");
});
test("formula library is integrated into the actual NERAL chat composer",()=>{
  const page=read("templates/neral_preview.html");
  const script=read("static/neral-formulas.js");
  assert.match(page,/id="formulaLibraryToggle"/);
  assert.match(page,/id="formulaLibraryPanel"/);
  assert.match(page,/id="formulaLibrarySearch"/);
  assert.match(page,/id="formulaLibraryCategory"/);
  assert.match(page,/\/static\/neral-formulas.js/);
  assert.match(page,/\/static\/neral-formulas.css/);
  assert.match(script,/setRangeText|insertText/);
  assert.match(script,/dispatchEvent/);
  assert.doesNotMatch(script,/innerHTML|outerHTML|\beval\s*\(|new Function|fetch\s*\(/);
});
test("mhchem extension ships offline before auto-rendering, both app shells",()=>{
  for(const page of ["templates/index.html","templates/neral_preview.html"]){
    const content=read(page);
    assert.ok(content.includes("/static/vendor/katex/mhchem.min.js"));
    assert.ok(content.indexOf("katex.min.js")<content.indexOf("mhchem.min.js"));
    assert.ok(content.indexOf("mhchem.min.js")<content.indexOf("auto-render.min.js"));
  }
  const vendor=read("scripts/vendor_katex.py");
  const build=read("scripts/build_neral_1_0_preview.py");
  assert.ok(vendor.includes("mhchem.min.js"));
  assert.ok(build.includes('"static/vendor/katex/mhchem.min.js"'));
  assert.ok(build.includes('"static/neral-formulas.js"'));
  assert.ok(build.includes('"static/neral-formulas.css"'));
});
test("app updater remains installed and checks on startup without a manual new EXE download",()=>{
  const source=read("neral_oneclick.py");
  assert.match(source,/start_update_check/);
  assert.match(source,/check_windows_release/);
  assert.match(source,/restart_into_update/);
  assert.match(source,/Update installieren und neu starten/);
  const workflow=read(".github/workflows/neral-single-click-windows.yml");
  assert.ok(workflow.includes("publish-windows:"));
  assert.ok(workflow.includes("'static/**'"));
});
