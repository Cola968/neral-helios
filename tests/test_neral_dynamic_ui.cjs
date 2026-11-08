"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const ui=require("../static/neral-dynamic-ui.js");

function title(kind){return {type:kind,title:"Demonstration"};}
test("dynamic intent identifies interactive tasks without affecting ordinary chat",()=>{
  assert.equal(ui.intentFor("Erstelle ein interaktives Quiz über Elektrizität"),"quiz");
  assert.equal(ui.intentFor("Zeig mir einen Rechner zur Prozentrechnung"),"calculator");
  assert.equal(ui.intentFor("Erstelle ein Balkendiagramm für diese Werte"),"chart");
  assert.equal(ui.intentFor("Simuliere einen schrägen Wurf"),"simulation");
  assert.equal(ui.intentFor("Wie heißt die Hauptstadt Deutschlands?"),null);
  assert.equal(ui.intentFor("Hallo"),null);
});
test("calculator expression evaluator handles math but never executes JS",()=>{
  const vars={u:12,r:4};
  assert.equal(ui.evaluate("u/r",vars),3);
  assert.equal(ui.evaluate("2^3 + 3*(4-2)",vars),14);
  assert.equal(ui.evaluate("sqrt(9)+abs(-2)",vars),5);
  assert.equal(ui.evaluate("-2^2",vars),-4);
  for(const bad of ["window.alert(1)","this.constructor.constructor('alert(1)')()","u=4","u;r","fetch('https://x')","1/0","2^10000000"]){
    assert.equal(ui.evaluate(bad,vars),null,bad);
  }
});
test("strict calculator schema rejects free code, invalid range and nonexistent symbols",()=>{
  const valid={...title("calculator"),inputs:[
    {id:"u",label:"Spannung",unit:"V",min:1,max:230,step:1,value:12},
    {id:"r",label:"Widerstand",unit:"Ω",min:1,max:100,step:1,value:4}
  ],expression:"u/r",resultLabel:"Strom",unit:"A"};
  assert.equal(ui.validate(valid).type,"calculator");
  assert.equal(ui.validate({...valid,expression:"window.alert(1)"}),null);
  assert.equal(ui.validate({...valid,expression:"x/r"}),null);
  assert.equal(ui.validate({...valid,inputs:[{...valid.inputs[0],step:0},valid.inputs[1]]}),null);
  assert.equal(ui.validate({...valid,inputs:[{...valid.inputs[0],max:1e30},valid.inputs[1]]}),null);
});
test("quiz, chart, table and simulation all have bounded declarative schemas",()=>{
  const quiz={...title("quiz"),questions:[{prompt:"Wie viel ist 2+2?",options:["3","4","5"],answer:1,explanation:"2+2=4"}]};
  const chart={...title("chart"),labels:["BER","HAM"],values:[4,2],unit:"Flüge"};
  const table={...title("table"),headers:["Flughafen","Flüge"],rows:[["BER","4"],["HAM","2"]]};
  const simulation={...title("simulation"),preset:"projectile"};
  for(const widget of [quiz,chart,table,simulation]){
    assert.equal(ui.validate(widget).type,widget.type);
  }
  assert.equal(ui.validate({...quiz,questions:[{...quiz.questions[0],answer:5}]}),null);
  assert.equal(ui.validate({...chart,values:[4,"alert(1)"]}),null);
  assert.equal(ui.validate({...table,rows:[["BER"]]}),null);
  assert.equal(ui.validate({...simulation,preset:"javascript"}),null);
  assert.equal(ui.validate({type:"html",title:"x",html:"<script>alert(1)</script>"}),null);
});
test("one valid neral-ui fenced block becomes a widget while text stays readable",()=>{
  const open=String.fromCharCode(96).repeat(3);
  const spec={...title("simulation"),preset:"projectile"};
  const result=ui.extract("Hier ist die Simulation.\n"+open+"neral-ui\n"+JSON.stringify(spec)+"\n"+open+"\nÄndere die Werte.");
  assert.equal(result.widget.type,"simulation");
  assert.match(result.text,/Hier ist die Simulation/);
  assert.match(result.text,/Ändere die Werte/);
  assert.doesNotMatch(result.text,/neral-ui/);
  const invalid=ui.extract("Erklärung\n"+open+"neral-ui\n{\"type\":\"html\"}\n"+open);
  assert.equal(invalid.widget,null);
  assert.match(invalid.text,/neral-ui/);
  assert.equal(ui.extract("Nur normaler Chat.").widget,null);
});
test("known physics questions can produce an honest local fallback widget",()=>{
  const widget=ui.fallbackFor("Zeige mir einen interaktiven Coulomb-Rechner");
  assert.equal(widget.type,"calculator");
  assert.ok(ui.evaluate(widget.expression,Object.fromEntries(widget.inputs.map(x=>[x.id,x.value])))>0);
  assert.equal(ui.fallbackFor("Wie spät ist es?"),null);
});
test("UI stays in the local release, exposes no executable content from model",()=>{
  const root=path.join(__dirname,"..");
  const html=fs.readFileSync(path.join(root,"templates/neral_preview.html"),"utf8");
  const build=fs.readFileSync(path.join(root,"scripts/build_neral_1_0_preview.py"),"utf8");
  const preview=fs.readFileSync(path.join(root,"static/neral-preview.js"),"utf8");
  const source=fs.readFileSync(path.join(root,"static/neral-dynamic-ui.js"),"utf8");
  assert.match(html,/neral-dynamic-ui\.js/);
  assert.match(build,/static\/neral-dynamic-ui\.js/);
  assert.match(preview,/NERALDynamicUI/);
  assert.doesNotMatch(source,/\b(?:eval|new Function)\s*\(/);
  assert.doesNotMatch(source,/\.innerHTML\s*=/);
});
