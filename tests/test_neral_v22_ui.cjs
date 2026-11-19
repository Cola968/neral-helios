"use strict";
const test=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const js=fs.readFileSync(path.join(__dirname,"../static/neral-preview.js"),"utf8");
test("explicit unverified Helios fallback tool status shows typed sandbox reason",()=>{
  assert.match(js,/Werkzeug: Python-Sandbox fehlgeschlagen/);
  for(const reason of ["docker_unavailable","invalid_python_code","runtime_error","empty_stdout","timeout"]){
    assert.ok(js.includes(reason),"missing stable error code: "+reason);
  }
  assert.match(js,/Modus: Helios \\(Modellantwort ungeprüft\\)/);
  assert.match(js,/result\.sandbox_executed===false/);
  assert.match(js,/result\.sandbox_failure_code/);
});
