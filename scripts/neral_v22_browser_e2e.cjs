/* V2.2 real Chrome CDP E2E: forced sandbox switch, actual Docker output. */
"use strict";
const fs=require("node:fs"),os=require("node:os"),path=require("node:path");
const {spawn}=require("node:child_process"),crypto=require("node:crypto");
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const base="http://127.0.0.1:9690/";
const out={status:"BLOCKED",real_browser:false,slider:[],v22:{hut:null,birthday:null},account_deleted:false,live_sources:0};
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

async function send(prompt,{web=false,kind,force=false}){
  stage="v2-"+kind;
  await choose("Luna");
  const active=await evaluate("(()=>{const x=document.getElementById('forceSandbox');"+
    "if(!x||x.disabled)throw Error('Force sandbox toggle unavailable');"+
    "x.checked="+(force?"true":"false")+";x.dispatchEvent(new Event('change',{bubbles:true}));"+
    "return x.checked;})()");
  check(active===force,"Force sandbox toggle did not apply");
  await evaluate("(()=>{const x=document.getElementById('useWebSearch');x.checked="+(web?"true":"false")+";x.dispatchEvent(new Event('change',{bubbles:true}));return x.checked})()");
  const original=await evaluate("document.querySelectorAll('.message.assistant:not(.neral-live-partial)').length");
  await evaluate("(()=>{const e=document.getElementById('prompt');e.value="+JSON.stringify(prompt)+
    ";e.dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('submit').click();return true})()");
  const result=await poll(async()=>{
    const x=await evaluate("(()=>{const a=[...document.querySelectorAll('.message.assistant:not(.neral-live-partial)')];"+
       "const n=a.at(-1);const badge=n?.querySelector('.neral-tool-status');"+
       "return {count:a.length,body:n?.innerText||'',status:badge?.textContent||'',"+
       "badge_visible:!!badge&&getComputedStyle(badge).display!=='none'&&getComputedStyle(badge).visibility!=='hidden',"+
       "source_links:[...n?.querySelectorAll('a.neral-source-link')||[]].map(x=>x.getAttribute('href')),"+
       "busy:document.getElementById('submit').disabled,error:document.getElementById('error')?.textContent||''};})()");
    if(x?.error&&!x.busy&&x.count<=original)throw Error("Chat UI returned error: "+String(x.error).slice(0,80));
    return x?.count>original&&!x.busy?x:null;
  },210000);
  check(result.badge_visible,"Tool-status line missing or hidden");
  check(!/Grenze:/i.test(result.body),"Unexpected canned Grenze text");
  return result;
}
async function net(type){
  return await poll(async()=>evaluate("(()=>{const a=window.__neralLiveNetwork||[];return a.filter(x=>x.type==="+JSON.stringify(type)+").at(-1)||null})()"),10000);
}
async function main(){
  try{
    stage="browser";await launch();
    stage="loaded";await poll(()=>evaluate("document.documentElement?.dataset?.neralThreeTier==='true'&&document.getElementById('modeEffortSlider')?.max==='2'"),45000);
    const session=await api("/api/neral/preview/session");
    check(session.three_tier_enabled===true,"Three-tier local service disabled");
    csrf=session.csrf;
    await evaluate("window.__e2e={csrf:"+JSON.stringify(csrf)+"};true");
    const username="e2ev22"+crypto.randomBytes(6).toString("hex");
    password="E2E-V22-"+crypto.randomBytes(18).toString("hex");
    stage="synthetic-account";
    const registered=await api("/api/neral/account/register","POST",{username,password});
    check(registered.authenticated===true,"Account registration failed");
    csrf=registered.csrf;
    await evaluate("window.__e2e.csrf="+JSON.stringify(csrf)+";"+
      "window.dispatchEvent(new CustomEvent('neral:account-csrf',{detail:{csrf:"+JSON.stringify(csrf)+"}}));"+
      "window.__neralLiveNetwork=[];"+
      "(()=>{const old=window.fetch.bind(window);window.fetch=async(...args)=>{"+
        "const route=String(args[0]||'');const r=await old(...args);"+
        "if(route.includes('/api/neral/chat/three-tier')||route.includes('/api/web/search')){"+
          "r.clone().json().then(d=>{window.__neralLiveNetwork.push("+
          "{type:route.includes('/api/web/search')?'search':'chat',status:r.status,"+
          "force_sandbox:typeof args[1]?.body==='string' && "+
          "(()=>{try{return JSON.parse(args[1].body).force_sandbox===true}catch{return false}})(),"+
          "query:typeof d.query==='string'?d.query:'',"+
          "external_request:d.external_request===true,"+
          "source_count:Array.isArray(d.results)?d.results.length:0,"+
          "source_https:Array.isArray(d.results)&&d.results.every(s=>/^https:\\/\\//.test(s.url||'')),"+
          "verified:d.verified,tool_used:d.tool_used,tier:d.tier,model_generated:d.model_generated,"+
          "sandbox_executed:d.sandbox_executed,sandbox_attempts:d.sandbox_attempts,"+
          "sandbox_failure_code:d.sandbox_failure_code||null,"+
          "sandbox_failure_exception:d.sandbox_failure_exception||null,"+
          "model_fallback_generated:d.model_fallback_generated===true,"+
          "verification_status:d.verification_status||null,"+
          "scope:d.verification_scope,safe_to_promote:d.safe_to_promote});}).catch(()=>{});"+
        "}return r;};})();true");
    out.account_created=true;
    try{
      const hut="Drei Freundinnen mieten gemeinsam eine Berghütte für 1.260 €. "+
        "Nora übernachtet dort die vollen 6 Nächte, Kim reist nach 4 Nächten wieder ab, "+
        "und Pia kommt erst zur Hälfte der Zeit dazu, bringt aber für ihre letzten "+
        "2 Nächte noch ihren Bruder mit. Alle Übernachtungsplätze kosten pro "+
        "Person und Nacht exakt gleich viel. Wie viel zahlt jede der drei "+
        "Freundinnen (Pia inklusive ihres Bruders)?";
      const first=await send(hut,{web:false,kind:"mountain-hut",force:false});
      const firstNetwork=await net("chat");
      out.v22.hut_http={
        http_status:firstNetwork.status,
        tier:firstNetwork.tier,
        tool_used:firstNetwork.tool_used||null,
        sandbox_executed:firstNetwork.sandbox_executed===true,
        sandbox_attempts:firstNetwork.sandbox_attempts||0,
        sandbox_failure_code:firstNetwork.sandbox_failure_code||null,
        sandbox_failure_exception:firstNetwork.sandbox_failure_exception||null,
        model_fallback_generated:firstNetwork.model_fallback_generated===true,
        verification_status:firstNetwork.verification_status||null,
        badge:first.status,
        has_all_expected_amounts:/504/.test(first.body)&&/336/.test(first.body)&&/420/.test(first.body)
      };
      check(firstNetwork.status===200 && firstNetwork.tool_used==="helios_code_sandbox" &&
            firstNetwork.tier==="Helios" && firstNetwork.verified===false &&
            firstNetwork.safe_to_promote===false,
            "Hut: real Helios sandbox HTTP result absent");
      check(first.status==="Werkzeug: Helios Code-Sandbox (ausgeführt)",
            "Hut: executed sandbox status missing");
      check(/504/.test(first.body)&&/336/.test(first.body)&&/420/.test(first.body),
            "Hut: precise 504/336/420 € not all visible");
      out.v22.hut={sandbox_executed:true,amounts:[504,336,420]};
      const birthday="Geburtstagsparadoxon: In einer Gruppe mit genau 23 Personen, "+
        "wie hoch ist die Wahrscheinlichkeit, dass mindestens zwei am gleichen "+
        "Kalendertag Geburtstag haben? Nimm 365 gleichwahrscheinliche "+
        "Geburtstage, keine Schaltjahre. Antwort in Prozent mit zwei Nachkommastellen.";
      const second=await send(birthday,{web:false,kind:"birthday-paradox",force:true});
      const secondNetwork=await poll(async()=>{
        const r=await net("chat");
        return r.force_sandbox===true?r:null;
      },12000);
      check(secondNetwork.status===200 && secondNetwork.tool_used==="helios_code_sandbox" &&
            secondNetwork.tier==="Helios" && secondNetwork.verified===false &&
            secondNetwork.safe_to_promote===false,
            "Birthday: real forced Helios sandbox HTTP result absent");
      check(second.status==="Werkzeug: Helios Code-Sandbox (ausgeführt)",
            "Birthday: executed sandbox status missing");
      check(/50[.,]73\s*%/.test(second.body),
            "Birthday: expected 50.73% not visible in actual browser reply");
      out.v22.birthday={sandbox_executed:true,percent:"50.73",force_sandbox:true};
      // UI-only fault injection: the actual two mathematical tasks above
      // MUST use real local Helios/Docker; this synthetic response tests that
      // a failed tool is visibly labeled and its fallback text is rendered.
      const fallbackFixture={
        text:"Helios-Antwort (ohne Sandbox, ungeprüft):\n2 plus 3 ergibt 5.",
        tier:"Helios",mode:"HELIOS",model:"qwen3.5:9b+neral-v2-full",
        auto_escalated_from:"Luna",model_generated:true,verified:false,
        safe_to_promote:false,verification_scope:"none",proof_backend:null,
        model_origin:"local_neral_lora_experimental",adapter_loaded:true,
        adapter_sha256:"a".repeat(64),runtime_tier_verified:true,
        sandbox_executed:false,sandbox_failure_code:"docker_unavailable",
        sandbox_failure_reason:"Docker nicht aktiv",sandbox_attempts:1,
        tool_used:null,model_fallback_generated:true
      };
      const syntheticPayload=JSON.stringify(JSON.stringify(fallbackFixture));
      await evaluate("(()=>{const prior=window.fetch;let injected=false;"+
        "window.fetch=async(...args)=>{const url=String(args[0]||'');"+
        "if(!injected&&url.includes('/api/neral/chat/three-tier')){"+
          "injected=true;window.fetch=prior;return new Response("+syntheticPayload+
          ",{status:200,headers:{'Content-Type':'application/json'}});}"+
        "return prior(...args)};return true;})()");
      const failed=await send("Was ergibt zwei plus drei?",{
        web:false,kind:"simulated-docker-unavailable-ui",force:true});
      check(failed.status.includes("Werkzeug: Python-Sandbox fehlgeschlagen") &&
            failed.status.includes("Docker nicht aktiv") &&
            failed.status.includes("Modellantwort ungeprüft"),
            "Synthetic browser failure status did not display precise Docker reason");
      check(failed.body.includes("Helios-Antwort (ohne Sandbox, ungeprüft)") &&
            failed.body.includes("2 plus 3 ergibt 5.") &&
            !failed.body.includes("Es liegt kein ausgeführtes Ergebnis vor"),
            "Synthetic browser failure did not render an honest fallback answer");
      out.v22.simulated_browser_fallback=true;
      out.status="V22_BROWSER_E2E_PASS";
    }finally{
      out.last_stage=stage;stage="synthetic-account-cleanup";
      try{
        const deleted=await api("/api/neral/account","DELETE",{password});
        out.account_deleted=deleted.deleted===true;
      }catch{out.account_deleted=false}
      if(!out.account_deleted)out.status="BLOCKED_ACCOUNT_CLEANUP";
    }
  }catch(e){
    out.status="V22_BROWSER_E2E_BLOCKED";
    out.failed_stage=out.last_stage||stage;
    out.error_type=e?.name||"Error";
    out.safe_error_hint=String(e?.message||"unknown").slice(0,145);
  }finally{
    try{ws?.close()}catch{}try{browser?.kill()}catch{}
    if(profile)try{fs.rmSync(profile,{recursive:true,force:true,maxRetries:3,retryDelay:200})}catch{}
    console.log("NERAL_V22_LIVE_BROWSER_SCALAR "+JSON.stringify(out));
  }
  process.exitCode=out.status==="V22_BROWSER_E2E_PASS"&&out.account_deleted?0:1;
}
main().catch(()=>process.exit(1));
