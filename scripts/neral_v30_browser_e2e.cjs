/* V3.0 REAL Chrome CDP: gold/page reader, Z3 plan, nested math, current prose. */
"use strict";
const fs=require("node:fs"),os=require("node:os"),path=require("node:path");
const {spawn}=require("node:child_process"),crypto=require("node:crypto");
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const base="http://127.0.0.1:9690/";
const out={status:"BLOCKED",real_browser:false,slider:[],v30:{},account_deleted:false};
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
          "verification_status:d.verification_status||null,v3_checker_passed:d.v3_checker_passed===true,v3_agent_iterations:d.v3_agent_iterations||0,"+
          "scope:d.verification_scope,safe_to_promote:d.safe_to_promote});}).catch(()=>{});"+
        "}return r;};})();true");
    out.account_created=true;
    try{
      // Capture the ACTUAL SSE result; a search card alone is not proof
      // that a webpage was downloaded and read.
      await evaluate("(()=>{const prior=window.fetch.bind(window);"+
        "window.__v3Reads=[];window.fetch=async(...args)=>{"+
        "const u=String(args[0]||'');const r=await prior(...args);"+
        "if(u.includes('/api/web/search/stream'))r.clone().text().then(raw=>{"+
        "const lines=raw.split(/\\r?\\n/);let result=null;"+
        "for(let s of lines){if(s.startsWith('data:'))s=s.slice(5).trim();"+
        "try{let e=JSON.parse(s);if(e.type==='result')result=e.result}catch{}}"+
        "if(result)window.__v3Reads.push({pages_read:result.pages_read||0,"+
        "read_scope:result.read_scope||null,previews:result.previews||[],"+
        "results:result.results||[]});}).catch(()=>{});return r;};return true;})()");
      const gold="Suche bitte aktuell im Web nach dem aktuellen Goldpreis in "+
        "Euro pro Feinunze und rechne mir aus, was ein 250-Gramm-Goldbarren "+
        "zum reinen Materialwert kostet (1 Feinunze = 31,1035 Gramm). "+
        "Nenne mir deine Quellen.";
      const goldUi=await send(gold,{web:true,kind:"v3-current-gold",force:false});
      const goldHttp=await net("chat");
      const goldRead=await poll(async()=>evaluate(
        "window.__v3Reads?.at(-1)||null"),15000);
      const actualPages=[...(goldRead.results||[]),...(goldRead.previews||[])]
        .filter(x=>x.read_status==="read"&&typeof x.content==="string"&&x.content.length);
      check(goldRead.pages_read>0&&actualPages.length>0,"Gold: no actual fetched page text");
      check(goldHttp.status===200&&goldHttp.sandbox_executed===true&&
        goldHttp.tool_used==="helios_code_sandbox","Gold: Python not executed");
      check(goldHttp.v3_checker_passed===true,
        "Gold: no successful V3 checker");
      check(goldUi.source_links.some(u=>/^https:\/\//.test(u||"")),
        "Gold: no real source link");
      check(!/Sandbox-Ausgabe[^\n]*\n0(?:[,.]0+)?(?:\s|$)/i.test(goldUi.body),
        "Gold: zero dummy was published as answer");
      const money=[...goldUi.body.matchAll(/([1-9][0-9.,]{1,14})\s*(?:€|Euro|EUR)/gi)]
        .map(x=>x[1]).filter(Boolean);
      check(money.length>0,"Gold: no positive euro amount in answer");
      out.v30={gold:{pages_read:goldRead.pages_read,sandbox:true,
        euro_amounts:money.slice(0,6),source_count:goldUi.source_links.length}};
      const schedule="Verteile drei unterschiedliche Schichten 0, 1 und 2 "+
        "auf Ada, Berta und Cora, jede Person bekommt genau eine Schicht. "+
        "Ada darf nicht Schicht 0, Berta darf nicht Schicht 2, "+
        "Cora darf nicht Schicht 1. Nenne eine gültige Zuordnung und "+
        "prüfe alle Bedingungen.";
      const planUi=await send(schedule,{web:false,kind:"v3-z3-scheduling",force:false});
      const planHttp=await net("chat");
      check(planHttp.status===200&&planHttp.sandbox_executed===true&&
        planHttp.v3_checker_passed===true,"Plan: Z3/validation not confirmed");
      check(/Ada/i.test(planUi.body)&&/Berta/i.test(planUi.body)&&
        /Cora/i.test(planUi.body),"Plan: missing participants");
      out.v30.schedule={sandbox:true,checker:true};
      const chain="Für einen Kurs werden drei gleiche Pakete zu je 28 Euro "+
        "gekauft. Zwei Pakete kosten jeweils 25 Prozent weniger als "+
        "der angegebene Preis, das dritte kostet den vollen Preis. "+
        "Anschließend wird ein Gutschein von 12 Euro abgezogen. "+
        "Auf den verbleibenden Betrag kommen 19 Prozent Steuer. "+
        "Wie hoch ist die abschließende Zahlung?";
      const mathUi=await send(chain,{web:false,kind:"v3-chained-arithmetic",force:false});
      const mathHttp=await net("chat");
      check(mathHttp.status===200&&mathHttp.sandbox_executed===true&&
        mathHttp.v3_checker_passed===true,"Multi-step: no executed/checked calculation");
      check(/69[,.]02/.test(mathUi.body),"Multi-step: expected 69.02 EUR missing");
      out.v30.multi_step={expected:"69.02",sandbox:true};
      const knowledge="Was sind zwei konkrete technische Neuerungen in "+
        "Python 3.14 laut der offiziellen Python-Dokumentation What's New? "+
        "Bitte recherchiere die Informationen im Web und nenne die URL.";
      const knownUi=await send(knowledge,{web:true,kind:"v3-current-knowledge",force:false});
      const knownHttp=await net("chat");
      const knownRead=await poll(async()=>evaluate(
        "window.__v3Reads?.at(-1)||null"),15000);
      const knownPages=[...(knownRead.results||[]),...(knownRead.previews||[])]
        .filter(x=>x.read_status==="read"&&x.content);
      check(knownRead.pages_read>0&&knownPages.length>0,
        "Knowledge: only snippets, no fetched article body");
      check(knownHttp.status===200&&knownHttp.v3_checker_passed===true,
        "Knowledge: current-source reasoning not checked");
      check(knownUi.source_links.length>0&&knownUi.body.length>100,
        "Knowledge: empty answer or no sources");
      out.v30.knowledge={pages_read:knownRead.pages_read,
        source_count:knownUi.source_links.length};
      out.status="V30_BROWSER_E2E_PASS";
    }finally{
      out.last_stage=stage;stage="synthetic-account-cleanup";
      try{
        const deleted=await api("/api/neral/account","DELETE",{password});
        out.account_deleted=deleted.deleted===true;
      }catch{out.account_deleted=false}
      if(!out.account_deleted)out.status="BLOCKED_ACCOUNT_CLEANUP";
    }
  }catch(e){
    out.status="V30_BROWSER_E2E_BLOCKED";
    out.failed_stage=out.last_stage||stage;
    out.error_type=e?.name||"Error";
    out.safe_error_hint=String(e?.message||"unknown").slice(0,145);
  }finally{
    try{ws?.close()}catch{}try{browser?.kill()}catch{}
    if(profile)try{fs.rmSync(profile,{recursive:true,force:true,maxRetries:3,retryDelay:200})}catch{}
    console.log("NERAL_V30_LIVE_BROWSER_SCALAR "+JSON.stringify(out));
  }
  process.exitCode=out.status==="V30_BROWSER_E2E_PASS"&&out.account_deleted?0:1;
}
main().catch(()=>process.exit(1));
