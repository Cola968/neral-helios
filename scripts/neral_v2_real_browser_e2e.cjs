/* Real headless Chrome/CDP E2E: all inputs are synthetic; delete test account. */
"use strict";
const fs=require("node:fs"),os=require("node:os"),path=require("node:path");
const {spawn}=require("node:child_process"),crypto=require("node:crypto");
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const base="http://127.0.0.1:9690/";
const out={status:"BLOCKED",real_browser:false,account_deleted:false,real_code_sandbox:false,tool_status:false};
let browser,ws,profile,csrf,password,stage="init",nextId=0;
const pending=new Map();
function check(ok,message){if(!ok)throw Error(message)}
async function poll(fn,limit=30000){let last;const end=Date.now()+limit;while(Date.now()<end){try{let v=await fn();if(v)return v;}catch(e){last=e;}await sleep(400);}throw last||Error("Timed out in "+stage);}
async function launch(){
  const exe=process.env.NERAL_CHROME_EXE;
  check(exe&&fs.existsSync(exe),"Chrome/Edge executable missing");
  profile=fs.mkdtempSync(path.join(os.tmpdir(),"neral-live-chrome-"));
  browser=spawn(exe,["--headless=new","--disable-gpu","--disable-extensions",
    "--disable-background-networking","--no-proxy-server","--no-first-run","--no-default-browser-check",
    "--remote-debugging-address=127.0.0.1","--remote-debugging-port=0",
    "--user-data-dir="+profile,"about:blank"],{stdio:"ignore",windowsHide:true});
  const file=path.join(profile,"DevToolsActivePort");
  const port=await poll(()=>fs.existsSync(file)&&Number(fs.readFileSync(file,"utf8").trim().split(/\r?\n/)[0]),20000);
  const resp=await fetch("http://127.0.0.1:"+port+"/json/new?about:blank",{method:"PUT"});
  check(resp.ok,"Chrome CDP create target failed");
  const tab=await resp.json();
  ws=new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{let t=setTimeout(()=>reject(Error("CDP websocket timeout")),15000);
    ws.addEventListener("open",()=>{clearTimeout(t);resolve()},{once:true});
    ws.addEventListener("error",()=>{clearTimeout(t);reject(Error("CDP websocket failed"))},{once:true});
  });
  ws.addEventListener("message",e=>{let m;try{m=JSON.parse(e.data)}catch{return}
    if(!m.id)return;const task=pending.get(m.id);if(!task)return;pending.delete(m.id);
    m.error?task.reject(Error("CDP command failed")):task.resolve(m.result);
  });
  out.real_browser=true;
  await command("Page.enable");await command("Runtime.enable");
  // Do not trust the /json/new query parameter as navigation proof.
  await sleep(500);
  const navigation=await command("Page.navigate",{url:base});
  out.nav_error=navigation.errorText||null;
  if(navigation.errorText)throw Error("Chrome rejected local navigation");
}
function command(method,params={}){
  return new Promise((resolve,reject)=>{
    const id=++nextId;
    let timeout=setTimeout(()=>{pending.delete(id);reject(Error("CDP timeout "+method))},45000);
    pending.set(id,{resolve:x=>{clearTimeout(timeout);resolve(x)},reject:e=>{clearTimeout(timeout);reject(e)}});
    ws.send(JSON.stringify({id,method,params}));
  });
}
async function evaluate(expression){
  let r=await command("Runtime.evaluate",{expression,awaitPromise:true,returnByValue:true});
  check(!r.exceptionDetails,"JavaScript exception at "+stage);
  return r.result?.value;
}
async function api(route,method="GET",body){
  const opts={method,credentials:"same-origin",headers:{"Content-Type":"application/json"}};
  if(body!==undefined)opts.body=JSON.stringify(body);
  const expr="(async()=>{const o="+JSON.stringify(opts)+";if(o.method!=='GET')o.headers['X-Neral-CSRF']=window.__e2e.csrf;"+
    "const r=await fetch("+JSON.stringify(route)+",o);const d=await r.json();return {status:r.status,data:d};})()";
  let res=await evaluate(expr);
  check(res.status<400,"HTTP "+res.status+" for "+route);
  return res.data;
}
async function choose(tier){
  const pos={Luna:"0",Terra:"1",Helios:"2"}[tier];check(pos!==undefined,"Unknown tier");
  const e="(()=>{const s=document.getElementById('modeEffortSlider');s.value="+JSON.stringify(pos)+
    ";s.dispatchEvent(new Event('input',{bubbles:true}));"+
    "return {label:document.getElementById('modeDropdownLabel').textContent,max:s.max};})()";
  let v=await evaluate(e);
  check(v&&v.label===tier&&v.max==="2","Slider mismatch for "+tier);
  out.slider.push(tier);
}

async function main(){
  try{
    stage="browser";await launch();
    stage="three-tier-ui";await poll(()=>evaluate("document.documentElement?.dataset?.neralThreeTier==='true'&&document.getElementById('modeEffortSlider')?.max==='2'"),45000);
    const session=await api("/api/neral/preview/session");
    check(session.three_tier_enabled===true,"Expected local three-tier runtime not enabled");
    csrf=session.csrf;
    await evaluate("window.__e2e={csrf:"+JSON.stringify(csrf)+"};true");
    const username="e2ev2"+crypto.randomBytes(7).toString("hex");
    password="E2E-V2-"+crypto.randomBytes(18).toString("hex");
    stage="synthetic-account";
    const registration=await api("/api/neral/account/register","POST",{username,password});
    check(registration.authenticated===true,"Synthetic local account registration failed");
    csrf=registration.csrf;
    await evaluate("window.__e2e.csrf="+JSON.stringify(csrf)+";"+
      "window.dispatchEvent(new CustomEvent('neral:account-csrf',{detail:{csrf:"+JSON.stringify(csrf)+"}}));"+
      "window.__e2eData=[];"+
      "(()=>{const original=window.fetch.bind(window);window.fetch=async(...args)=>{"+
      "const path=String(args[0]||'');const response=await original(...args);"+
      "if(path.includes('/api/neral/chat/three-tier')){"+
        "response.clone().json().then(d=>window.__e2eData.push({"+
          "status:response.status,tool_used:d.tool_used,sandbox_executed:d.sandbox_executed,"+
          "verified:d.verified,safe_to_promote:d.safe_to_promote,tier:d.tier,"+
          "model_generated:d.model_generated,auto_escalated_from:d.auto_escalated_from"+
        "})).catch(()=>{});"+
      "}return response;};})();true");
    out.account_created=true;
    try{
      stage="real-free-code";
      await choose("Luna");
      const before=await evaluate("document.querySelectorAll('.message.assistant:not(.neral-live-partial)').length");
      await evaluate("(()=>{const w=document.getElementById('useWebSearch');w.checked=false;w.dispatchEvent(new Event('change',{bubbles:true}));"+
        "const e=document.getElementById('prompt');e.value='Berechne die exakte Summe von 26, 34 und 18.';"+
        "e.dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('submit').click();return true})()");
      const ui=await poll(async()=>{
        const result=await evaluate("(()=>{const a=[...document.querySelectorAll('.message.assistant:not(.neral-live-partial)')];"+
          "const n=a.at(-1), badge=n?.querySelector('.neral-tool-status');"+
          "return {count:a.length,status:badge?.textContent||'',"+
          "status_visible:!!badge&&getComputedStyle(badge).display!=='none'&&getComputedStyle(badge).visibility!=='hidden',"+
          "body:n?.innerText||'',header:n?.querySelector('strong')?.textContent||'',"+
          "error:document.getElementById('error')?.textContent||'',"+
          "busy:document.getElementById('submit').disabled};})()");
        if(result?.error&&!result.busy&&result.count<=before)throw Error("Real browser chat error");
        return result?.count>before&&!result.busy?result:null;
      },190000);
      check(ui.status_visible&&ui.status==="Werkzeug: Helios Code-Sandbox (ausgeführt)","Real browser tool status missing or misleading");
      check(ui.header==="NERAL","UI identity heading not preserved");
      check(/Sandbox-Ausgabe \(nicht unabhängig geprüft\):/.test(ui.body)&&/\b78\b/.test(ui.body),"No actual expected model-generated computation output");
      check(!/Grenze:/i.test(ui.body),"Old canned disclaimer appeared");
      const meta=await poll(()=>evaluate("window.__e2eData.at(-1)||null"),10000);
      check(meta.status===200&&meta.sandbox_executed===true&&meta.tool_used==="helios_code_sandbox"&&
            meta.verified===false&&meta.safe_to_promote===false&&
            meta.tier==="Helios"&&meta.model_generated===true&&meta.auto_escalated_from==="Luna",
            "Real three-tier HTTP returned incorrect execution/proof metadata");
      out.real_code_sandbox=true;out.tool_status=true;
      out.status="V2_BROWSER_SANDBOX_PASS";
    }finally{
      out.last_stage=stage;stage="synthetic-account-delete";
      try{
        const deletion=await api("/api/neral/account","DELETE",{password});
        out.account_deleted=deletion.deleted===true;
      }catch{out.account_deleted=false}
      if(!out.account_deleted)out.status="BLOCKED_ACCOUNT_CLEANUP";
    }
  }catch(e){
    out.status="V2_BROWSER_SANDBOX_BLOCKED";
    out.failed_stage=out.last_stage||stage;
    out.error_type=e?.name||"Error";
    out.safe_error_hint=String(e?.message||"unknown").slice(0,145);
  }finally{
    try{ws?.close()}catch{}try{browser?.kill()}catch{}
    if(profile)try{fs.rmSync(profile,{recursive:true,force:true,maxRetries:3,retryDelay:200})}catch{}
    console.log("NERAL_V2_REAL_BROWSER_SCALAR "+JSON.stringify(out));
  }
  process.exitCode=out.status==="V2_BROWSER_SANDBOX_PASS"&&out.account_deleted?0:1;
}
main().catch(()=>process.exit(1));
