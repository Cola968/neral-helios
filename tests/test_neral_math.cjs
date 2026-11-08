"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const root=path.resolve(__dirname,".."),read=p=>fs.readFileSync(path.join(root,p),"utf8");

function boot(){
  const calls=[];
  const mathApi={renderMathInElement:(node,options)=>{
    calls.push({node,options});
  }};
  const context={
    window:{renderMathInElement:mathApi.renderMathInElement},
    document:{},
    console
  };
  vm.runInNewContext(read("static/neral-math.js"),context,{timeout:1500});
  return {api:context.window.NeralMath,calls};
}

test("normal assistant formulas are typeset locally with safe delimiters",()=>{
  const {api,calls}=boot();
  const node={tagName:"DIV"};
  assert.equal(api.render(node),true);
  assert.equal(calls.length,1);
  const delimiters=calls[0].options.delimiters;
  assert.ok(delimiters.some(x=>x.left==="$$"&&x.right==="$$"&&x.display===true));
  assert.ok(delimiters.some(x=>x.left==="$"&&x.right==="$"&&x.display===false));
  assert.ok(delimiters.some(x=>x.left==="\\["&&x.right==="\\]"&&x.display===true));
  assert.ok(delimiters.some(x=>x.left==="\\("&&x.right==="\\)"&&x.display===false));
  assert.equal(calls[0].options.trust,false);
  assert.equal(calls[0].options.throwOnError,false);
  assert.ok(calls[0].options.ignoredTags.includes("code"));
  assert.ok(calls[0].options.ignoredTags.includes("pre"));
  assert.ok(calls[0].options.ignoredClasses.includes("katex"));
});
test("missing KaTeX fails without changing answer text or fetching CDN",()=>{
  const window={};
  vm.runInNewContext(read("static/neral-math.js"),{window,document:{},console},{timeout:1500});
  const text={tagName:"P",textContent:"$E=mc^2$"};
  assert.equal(window.NeralMath.render(text),false);
  assert.equal(text.textContent,"$E=mc^2$");
  assert.doesNotMatch(read("static/neral-math.js"),/innerHTML|outerHTML|fetch\s*\(|eval\s*\(/);
});
test("code blocks and user text are never passed to math auto-renderer",()=>{
  const {api,calls}=boot();
  assert.equal(api.render({tagName:"CODE"}),false);
  assert.equal(api.render({tagName:"PRE"}),false);
  assert.equal(api.render({tagName:"TEXTAREA"}),false);
  assert.equal(calls.length,0);
});
test("NERAL repairs only the exact known lost-fraction Kepler math-fence forms",()=>{
  const {api}=boot();
  const dual="T2a3=4π2G(M+m)";
  const one="T2a3=4π2GM";
  const two=api.normalizeMathFence(dual);
  const heavy=api.normalizeMathFence(one);
  assert.equal(two,String.raw`\frac{T^2}{a^3}=\frac{4\pi^2}{G(M+m)}`);
  assert.equal(heavy,String.raw`\frac{T^2}{a^3}=\frac{4\pi^2}{GM}`);
  assert.equal(api.normalizeMathFence(String.raw`\frac{T^2}{a^3}=\frac{4\pi^2}{G(M+m)}`),two);
  assert.equal(api.normalizeMathFence("T2a3=4π2/(G(M+m))"),null,
    "unknown or ambiguous corrupt expressions must never be guessed");
  const katex=require(path.join(root,"static/vendor/katex/katex.min.js"));
  const rendered=katex.renderToString(two,{displayMode:true,throwOnError:true,trust:false});
  assert.match(rendered,/mfrac/);
  assert.match(rendered,/katex-display/);
  assert.doesNotMatch(rendered,/katex-error/);
});
test("a fenced math block is rendered by local KaTeX as a proper fraction, not code",()=>{
  const {api}=boot();
  const invocations=[];
  const window={katex:{render:(formula,node,options)=>invocations.push({formula,node,options})}};
  const sandbox={window,document:{},console};
  vm.runInNewContext(read("static/neral-math.js"),sandbox,{timeout:1500});
  const node={tagName:"DIV"};
  assert.equal(window.NeralMath.renderBlock(node,"T2a3=4π2G(M+m)"),true);
  assert.equal(invocations.length,1);
  assert.equal(invocations[0].formula,String.raw`\frac{T^2}{a^3}=\frac{4\pi^2}{G(M+m)}`);
  assert.equal(invocations[0].options.trust,false);
  assert.equal(invocations[0].options.throwOnError,true);
  assert.equal(invocations[0].options.displayMode,true);
  assert.equal(window.NeralMath.renderBlock(node,"<img src=x onerror=alert(1)>"),false);
  assert.equal(window.NeralMath.renderBlock(node,""),false);
  assert.equal(invocations.length,1);
});
test("fenced math is recognized separately from code while original answer is preserved",()=>{
  const reference=read("static/neral-reference-ui.js");
  assert.match(reference,/math\|latex\|tex/);
  assert.match(reference,/NeralMath\?\.renderBlock/);
  assert.match(reference,/reference-math/);
  assert.match(reference,/reference-code/);
  assert.doesNotMatch(reference,/\.innerHTML\s*=/);
});
test("real offline KaTeX engine typesets relativity as MathML and HTML",()=>{
  const katex=require(path.join(root,"static/vendor/katex/katex.min.js"));
  const bs=String.fromCharCode(92);
  const examples=["E=mc^2",bs+"Delta t="+bs+"gamma"+bs+"Delta t_0",
    "G_{"+bs+"mu"+bs+"nu}="+bs+"frac{8"+bs+"pi G}{c^4}T_{"+bs+"mu"+bs+"nu}",
    "r_s="+bs+"frac{2GM}{c^2}"];
  for(const expression of examples){
    const html=katex.renderToString(expression,{
      throwOnError:true,trust:false,output:"htmlAndMathml"
    });
    assert.match(html,/class="katex"/);
    assert.match(html,/<math[\s>]/);
    assert.match(html,/katex-html/);
  }
});
test("code fences are converted to safe code DOM before mathematical formatting",()=>{
  const preview=read("static/neral-preview.js");
  const reference=read("static/neral-reference-ui.js");
  assert.match(preview,/text\.includes\(String\.fromCharCode\(96\)\.repeat\(3\)\)/);
  assert.match(reference,/NeralMath\?\.render\(result\)/);
});
test("real preview, restored messages, dynamic tables and original Helios chat are wired",()=>{
  const preview=read("static/neral-preview.js");
  const dynamic=read("static/neral-dynamic-ui.js");
  const helios=read("static/helios-core.js");
  const local=read("static/neral-local-chat.js");
  assert.match(preview,/NeralMath/);
  assert.match(preview,/NeralMath\?\.render\(body\)/);
  assert.match(dynamic,/NeralMath/);
  assert.match(dynamic,/table/);
  assert.match(helios,/NeralMath/);
  assert.match(local,/NeralMath/);
});
test("both local app shells use offline bundled KaTeX before other NERAL modules",()=>{
  for(const htmlFile of ["templates/neral_preview.html","templates/index.html"]){
    const html=read(htmlFile);
    for(const asset of [
      "/static/vendor/katex/katex.min.css",
      "/static/vendor/katex/katex.min.js",
      "/static/vendor/katex/auto-render.min.js",
      "/static/neral-math.js",
      "/static/neral-math.css"
    ])assert.ok(html.includes(asset),htmlFile+" missing "+asset);
    assert.ok(html.indexOf("katex.min.js")<html.indexOf("auto-render.min.js"));
    assert.ok(html.indexOf("auto-render.min.js")<html.indexOf("neral-math.js"));
    assert.doesNotMatch(html,/https?:\/\/(?:cdn\.jsdelivr\.net|unpkg\.com)/i);
  }
});
test("math remains available offline in Windows build and verified release ZIP",()=>{
  const pack=read("scripts/build_neral_1_0_preview.py");
  const vendor=read("scripts/vendor_katex.py");
  const workflow=read(".github/workflows/neral-single-click-windows.yml");
  const rc=read(".github/workflows/neral-1-0-release-rc.yml");
  assert.ok(pack.includes('"static/vendor/katex/katex.min.js"'));
  assert.ok(pack.includes('"static/neral-math.js"'));
  assert.ok(pack.includes('"static/vendor/katex/fonts/'));
  assert.ok(vendor.includes("0.19.1"));
  assert.ok(workflow.includes("scripts/vendor_katex.py"));
  assert.ok(rc.includes("scripts/vendor_katex.py"));
});
test("rendered scientific table replaces only its visible Markdown duplication, retains original copy",()=>{
  const preview=read("static/neral-preview.js");
  assert.match(preview,/NeralMath\.hideRedundantMarkdownTable/);
  assert.match(preview,/ui\.mount/);
  const {api}=boot();
  const original=[
    "Diese Formeln sind wichtig:",
    "| Thema | Formel |",
    "| --- | --- |",
    "| Energie | $E=mc^2$ |",
    "| Radius | $r_s=\\frac{2GM}{c^2}$ |",
    "",
    "Soll ich ein Beispiel rechnen?"
  ].join("\n");
  const output=api.stripDuplicateTable(original);
  assert.match(output,/Diese Formeln sind wichtig/);
  assert.match(output,/Soll ich ein Beispiel rechnen/);
  assert.doesNotMatch(output,/^\|/m);
  assert.match(original,/\\frac/); // source remains untouched for copy.
});
