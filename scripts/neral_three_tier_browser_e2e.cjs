/* Real headless Chrome/CDP E2E: all inputs are synthetic; delete test account. */
"use strict";
const fs=require("node:fs"),os=require("node:os"),path=require("node:path");
const {spawn}=require("node:child_process"),crypto=require("node:crypto");
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const base="http://127.0.0.1:9690/";
const out={status:"BLOCKED",real_browser:false,slider:[],models:[],memory:false,rag:false,web:false,web_model_answer:false,account_deleted:false,everyday_money:false,everyday_z3:false};
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
async function send(tier,prompt,kind=""){
  stage="chat-"+tier+"-"+kind;
  await choose(tier);
  const original=await evaluate("document.querySelectorAll('.message.assistant:not(.neral-live-partial)').length");
  await evaluate("(()=>{const e=document.getElementById('prompt');e.value="+JSON.stringify(prompt)+
    ";e.dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('submit').click();return true})()");
  const res=await poll(async()=>{
    const value=await evaluate("(()=>{const a=[...document.querySelectorAll('.message.assistant:not(.neral-live-partial)')];"+
      "const n=a.at(-1);const h=n?.querySelector('strong');return {count:a.length,"+
      "body:n?.innerText||'',badge:h?.getAttribute('aria-label')||h?.title||'',"+
      "busy:document.getElementById('submit').disabled,error:document.getElementById('error')?.textContent||''};})()");
    if(value?.error&&!value.busy&&value.count<=original){
      out.ui_error_hint=String(value.error).slice(0,180);
      throw Error("Chat UI returned error");
    }
    return value?.count>original&&!value.busy?value:null;
  },tier==="Helios"?175000:130000);
  check(res.body.trim(),"Empty assistant response");
  if(kind==="proof")check(/Verifiziert/i.test(res.badge)&&/z3_unsat|trusted_source_arithmetic|independent_enumeration/.test(res.badge),
                           "Independent proof badge missing");
  if(kind==="everyday-money"){
    check(/NERAL Terra \(automatisch\).*Verifiziert.*trusted_source_arithmetic/.test(res.badge),
          "Luna did not automatically route math to verified Terra/Python");
    check(res.body.includes("3.226,89 €")&&res.body.includes("613,11 €")&&
          res.body.includes("1.613,44 €")&&res.body.includes("968,07 €")&&
          res.body.includes("645,38 €"),"Exact cents or three phase allocations missing");
    check(!/Grenze:|liefern\.Rechnung:/.test(res.body),"Misleading model math disclaimer");
    out.everyday_money=true;
  }
  if(kind==="everyday-z3"){
    check(/NERAL Helios \(automatisch\).*Verifiziert.*z3_sat_witness/.test(res.badge),
          "Luna did not automatically route schedule to verified Z3");
    check(/Person 1/.test(res.body)&&/Person 2/.test(res.body)&&
          /Gesamtdauer:/.test(res.body)&&
          ["A","B","C","D"].every(n=>new RegExp("\\b"+n+"\\s+\\d+[–-]\\d+").test(res.body)),
          "Z3 schedule has no complete, concrete task windows");
    check(!/Grenze:|liefern\.Plan:/.test(res.body),"Misleading model schedule disclaimer");
    out.everyday_z3=true;
  }
  if(kind==="memory")check(/ganymed[- ]?47/i.test(res.body),"Saved SQLite fact not recalled");
  if(kind==="rag")check(/17/.test(res.body)&&/e2eber|quelle|#c\d/i.test(res.body),"RAG answer missing value/citation");
  return res;
}
async function main(){
  try{
    stage="real-browser";await launch();
    stage="app-ui-loaded";await poll(()=>evaluate("document.documentElement?.dataset?.neralThreeTier==='true'&&document.getElementById('modeEffortSlider')?.max==='2'"),40000);
    stage="local-session";
    const session=await api("/api/neral/preview/session");
    check(session.three_tier_enabled===true,"Three tier preview disabled");
    csrf=session.csrf;
    await evaluate("window.__e2e={csrf:"+JSON.stringify(csrf)+"};true");
    const user="e2e"+crypto.randomBytes(6).toString("hex");
    password="E2E-"+crypto.randomBytes(18).toString("hex");
    stage="test-account";
    const registered=await api("/api/neral/account/register","POST",{username:user,password});
    check(registered.authenticated===true,"Account registration failed");
    csrf=registered.csrf;
    await evaluate("window.__e2e.csrf="+JSON.stringify(csrf)+";"+
      "window.dispatchEvent(new CustomEvent('neral:account-csrf',{detail:{csrf:"+JSON.stringify(csrf)+"}}));true");
    out.account_created=true;
    try{
      stage="memory-save";
      await api("/api/neral/cpu/memory","POST",{key:"e2e-testflugzeug",value:"Das E2E-Testflugzeug heißt Ganymed-47.",kind:"project",consent:true});
      const mem=await api("/api/neral/cpu/memory");
      check(JSON.stringify(mem).includes("Ganymed-47"),"Memory SQLite persistence failed");
      stage="document-upload";
      await api("/api/neral/cpu/documents","POST",{doc_id:"e2eber",source:"e2e-ber-test.md",
        text:("Das Prüfgerät E2E-BER-73 muss alle 17 Tage kontrolliert werden. Referenz E2E-BER-QUELLE. ").repeat(14)});
      const documents=await api("/api/neral/cpu/documents/search?q=E2E-BER-73");
      check(JSON.stringify(documents).includes("e2e-ber-test.md"),"RAG SQLite index failed");
      await send("Luna","Antworte mit einem kurzen Hallo.");
      out.models.push("Luna");
      await send("Luna","Wie heißt das E2E-Testflugzeug aus meinen gespeicherten Angaben?","memory");
      out.memory=true;
      await send("Terra","How many positive divisors does the integer 120 have?","proof");
      out.models.push("Terra");
      await send("Terra","Was ist laut dem Dokument e2e-ber-test.md das Wartungsintervall von Prüfgerät E2E-BER-73? Zitiere die Quellenkennung.","rag");
      out.rag=true;
      await send("Helios","Color vertices 0..5 with red, green or blue. Vertex 0 is red, and endpoints of edges [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 4), (1, 5)] must differ. How many valid colorings?","proof");
      out.models.push("Helios");
      // The EXACT two real user prompts, with the slider LEFT on Luna.
      await send("Luna","Teile 3.840 € brutto (inkl. 19 % MwSt.) exakt in Netto und Steuer auf und verteile den Nettobetrag im Verhältnis 5 : 3 : 2 auf drei Projektphasen.","everyday-money");
      await send("Luna","Vier Aufgaben A (3 Std.), B (4 Std.), C (2 Std.) und D (5 Std.) müssen auf zwei Personen verteilt werden. A muss vor B abgeschlossen sein, B und C dürfen nicht gleichzeitig laufen, und die Gesamtdauer darf 8 Stunden nicht überschreiten. Erstelle einen gültigen Zeitplan.","everyday-z3");
      stage="live-web-consent";
      await evaluate("(()=>{const w=document.getElementById('useWebSearch');w.checked=true;"+
        "w.dispatchEvent(new Event('change',{bubbles:true}));return w.checked;})()");
      const webResponse=await send("Luna","Suche im Internet nach aktuellen BER-Flugverbindungen.");
      const links=await evaluate("document.querySelectorAll('.neral-source-link').length");
      check(links>0,"Real browser did not render search sources");
      // Previously E2E accepted source-only fallback as a successful search.
      const web_model_answer = webResponse.body.trim().length>45 &&
        /BER|Brandenburg|Flug|Airport/i.test(webResponse.body) &&
        !/Suchtreffer ohne Modellzusammenfassung|NERAL Websuche/.test(webResponse.badge) &&
        !/Die lokale Websuche hat \\d+|Die Websuche hat \\d+/.test(webResponse.body) &&
        !/Modellzusammenfassung konnte nicht erzeugt werden/i.test(webResponse.body);
      out.web_model_answer=web_model_answer;
      out.live_search_links=links;
      check(web_model_answer,"Web search has links but no usable model answer");
      out.web=true;
      out.status="BROWSER_E2E_PASS";
    }finally{
      out.last_active_stage=stage;
      stage="delete-synthetic-account";
      try{const deletion=await api("/api/neral/account","DELETE",{password});
        out.account_deleted=deletion.deleted===true;}catch{out.account_deleted=false}
      if(!out.account_deleted)out.status="BLOCKED_ACCOUNT_CLEANUP";
    }
  }catch(e){
    out.status="BROWSER_E2E_BLOCKED";out.failed_stage=out.last_active_stage||stage;out.error_type=e?.name||"Error";out.safe_error_hint=String(e?.message||"unknown").slice(0,135);
    try{
      const diagnostic=await evaluate("(async()=>{const s=await fetch('/api/neral/preview/session').then(r=>r.json()).catch(()=>({}));return {"+
        "on_expected_url:location.origin==='http://127.0.0.1:9690',location_scheme:location.protocol,location_host:location.host,"+
        "ready_state:document.readyState,slider_exists:!!document.getElementById('modeEffortSlider'),"+
        "slider_max:document.getElementById('modeEffortSlider')?.max||null,"+
        "tier_dataset:document.documentElement?.dataset?.neralThreeTier||null,"+
        "script_count:document.scripts?.length||0,session_three_tier:s.three_tier_enabled===true,"+
        "session_tier_count:Array.isArray(s.three_tier_levels)?s.three_tier_levels.length:0};})()");
      out.boot_diagnostic=diagnostic;
    }catch{out.boot_diagnostic="CDP_DIAGNOSTIC_UNAVAILABLE"}
  }
  finally{try{ws?.close()}catch{}try{browser?.kill()}catch{}
    if(profile)try{fs.rmSync(profile,{recursive:true,force:true,maxRetries:3,retryDelay:200})}catch{}
    console.log("NERAL_THREE_TIER_BROWSER_SCALAR "+JSON.stringify(out));
  }
  process.exitCode=out.status==="BROWSER_E2E_PASS"?0:1;
}
main().catch(()=>process.exit(1));
