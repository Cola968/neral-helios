/* NERAL V3 PR567 actual Chrome on 127.0.0.1:9690, four independent prompts in one chat. */
"use strict";
const fs=require("node:fs"),os=require("node:os"),path=require("node:path");
const {spawn}=require("node:child_process"),crypto=require("node:crypto");
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const base="http://127.0.0.1:9690/";
const out={status:"BLOCKED",real_browser:false,slider:[],v3:{gold:null,z3:null,text:null,firefox:null},account_deleted:false};
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
  stage="v3-"+kind;
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
  },480000);
  check(result.badge_visible,"Tool-status line missing or hidden");
  check(!/Grenze:/i.test(result.body),"Unexpected canned Grenze text");
  return result;
}
async function net(type){
  return await poll(async()=>evaluate("(()=>{const a=window.__neralLiveNetwork||[];return a.filter(x=>x.type==="+JSON.stringify(type)+").at(-1)||null})()"),10000);
}

async function main(){
  try {
    stage="launch";await launch();
    stage="homepage";await poll(()=>evaluate(
      "document.documentElement?.dataset?.neralThreeTier==='true'&&"+
      "document.getElementById('modeEffortSlider')?.max==='2'"),45000);
    const session=await api("/api/neral/preview/session");
    check(session.three_tier_enabled===true && session.evidence_agent_enabled===true,
      "V3 evidence agent not actually enabled on port 9690");
    csrf=session.csrf;
    await evaluate("window.__e2e={csrf:"+JSON.stringify(csrf)+"};true");
    const username="v3pr567"+crypto.randomBytes(6).toString("hex");
    password="V3-E2E-"+crypto.randomBytes(20).toString("hex");
    stage="temporary-account";
    const account=await api("/api/neral/account/register","POST",{username,password});
    check(account.authenticated===true,"Synthetic account registration failed");
    csrf=account.csrf;
    await evaluate("window.__e2e.csrf="+JSON.stringify(csrf)+";"+
      "window.dispatchEvent(new CustomEvent('neral:account-csrf',{detail:{csrf:"+JSON.stringify(csrf)+"}}));"+
      "window.__neralV3Calls=[];"+
      "(()=>{const old=window.fetch.bind(window);window.fetch=async(...args)=>{"+
        "const route=String(args[0]||'');const response=await old(...args);"+
        "if(route.includes('/api/neral/chat/three-tier')){"+
          "let historyCount=0;"+
          "try{const b=JSON.parse(args[1]?.body||'{}');historyCount=Array.isArray(b.history)?b.history.length:0}catch{}"+
          "response.clone().json().then(d=>{window.__neralV3Calls.push({"+
            "status:response.status,history_count:historyCount,"+
            "agent_checked:d.agent_checked===true,"+
            "agent_steps:Array.isArray(d.agent_steps)?d.agent_steps:[],"+
            "tool_used:d.tool_used||null,"+
            "page_read_count:Array.isArray(d.sources)?d.sources.filter(s=>s.page_read===true).length:0,"+
            "source_hosts:Array.isArray(d.sources)?d.sources.map(s=>{try{return new URL(s.url).hostname}catch{return ''}}):[],"+
            "verified:d.verified===true,agent_attempts:d.agent_attempts||0,"+
            "source_citation:typeof d.text==='string'&&/\\[web\\d+\\]/i.test(d.text),"+
            "has_z3_line:typeof d.text==='string'&&/Z3:/i.test(d.text),"+
            "has_missing_source:typeof d.text==='string'&&/Keine lesbare Webseite|Kein belegter Wert/i.test(d.text),"+
            "has_64:typeof d.text==='string'&&/\\b64\\b/.test(d.text),"+
            "has_4_litre:typeof d.text==='string'&&/\\b4(?:[,.]0+)?\\s*Liter/i.test(d.text),"+
            "has_firefox:typeof d.text==='string'&&/Firefox/i.test(d.text),"+
            "has_chrome:typeof d.text==='string'&&/Chrome/i.test(d.text)"+
          "});}).catch(()=>{});"+
        "}return response;};})();true");
    out.account_created=true;
    async function lastCall(){
      return await poll(()=>evaluate(
        "window.__neralV3Calls&&window.__neralV3Calls.at(-1)||null"),15000);
    }
    async function inspectTurn(label,prompt,{web=false,force=false}={}){
      const before=await evaluate("window.__neralV3Calls.length");
      const ui=await send(prompt,{web,force,kind:label});
      const network=await poll(async()=>{
        const data=await evaluate("window.__neralV3Calls.slice("+before+")");
        return data?.length?data.at(-1):null;
      },20000);
      check(network.status===200,label+": real local API HTTP "+network.status);
      check(network.agent_checked===true,label+": agent checker rejected real answer");
      check(network.verified===false,label+": model checker incorrectly promoted to proof");
      return {ui,network};
    }
    try{
      stage="v3-gold";
      const gold=await inspectTurn("gold",
        "Suche bitte aktuell im Web nach dem aktuellen Goldpreis in Euro pro Feinunze und rechne mir aus, "+
        "was ein 250-Gramm-Goldbarren zum reinen Materialwert kostet "+
        "(1 Feinunze = 31,1035 Gramm). Nenne mir deine Quellen.",{web:true});
      check(gold.network.tool_used==="python_sandbox","Gold: no executed Python tool");
      check(gold.network.page_read_count>=1,"Gold: no genuine read HTTPS page");
      check(!gold.network.has_missing_source,"Gold: price unavailable");
      check(gold.ui.source_links.length>=1,"Gold: actual HTTPS citations missing");
      const positiveGold=gold.ui.body.match(/(?:[1-9]\d{1,2}(?:[.,]\d{3})+|\d{4,6})[,.]\d{2}\s*(?:€|EUR)/);
      check(Boolean(positiveGold),"Gold: no positive material value");
      out.v3.gold={checked:true,real_page_count:gold.network.page_read_count,
        tool:"python_sandbox",positive_value:true};

      // No new-chat action: the next task must carry the previous gold turns
      // in the UI history while the planner and checker use ONLY its own rules.
      stage="v3-z3-same-chat";
      const shift=await inspectTurn("z3-same-chat",
        "Neue eigenständige Aufgabe: Drei Mitarbeitende Ada, Ben und Cem müssen "+
        "jeweils genau einen Tag von Montag, Dienstag und Mittwoch übernehmen. "+
        "An jedem Tag arbeitet genau eine Person. Ada muss Montag arbeiten. "+
        "Ben darf nicht Dienstag arbeiten. Cem darf nicht Mittwoch arbeiten. "+
        "Finde eine vollständige zulässige Schichtzuteilung und prüfe jede Bedingung.",
        {web:false});
      check(shift.network.history_count>=2,
        "Z3: test was not performed inside the existing chat history");
      check(shift.network.tool_used==="z3_solver",
        "Z3: wrong tool, expected actual constraint solver");
      check(shift.network.has_z3_line,
        "Z3: no real solver assignment in response");
      check(/Ada/i.test(shift.ui.body)&&/Ben/i.test(shift.ui.body)&&/Cem/i.test(shift.ui.body),
        "Z3: all three staff members must be assigned");
      out.v3.z3={checked:true,same_chat_history:shift.network.history_count,
        tool:"z3_solver",solver_output:true};

      stage="v3-nested-text";
      const nested=await inspectTurn("multi-step-inventory",
        "Jan stellt 4 Kisten mit je 18 Gläsern in den Raum. "+
        "8 Gläser gehen kaputt. In jedes übrige Glas füllt er 0,25 Liter Saft. "+
        "Der Behälter enthält vorher 20 Liter. Was bleibt im Behälter? "+
        "Wie viele gefüllte Gläser stehen bereit?",{web:false});
      check(nested.network.tool_used==="python_sandbox",
        "Inventory: actual Python tool not executed");
      check(nested.network.has_64&&nested.network.has_4_litre,
        "Inventory: correct 64 glasses and 4 litres missing");
      out.v3.text={checked:true,tool:"python_sandbox",
        filled_glasses:64,remaining_litres:4};

      stage="v3-firefox-comparison";
      const firefox=await inspectTurn("firefox-web-comparison",
        "Suche aktuelle Webseiten und vergleiche Mozilla Firefox mit Google Chrome: "+
        "Welche Browser-Engine verwenden sie und welche Unterschiede werden "+
        "beim Datenschutz und beim Schutz vor Tracking beschrieben? "+
        "Nenne belegbare Quellen für beide Browser.",{web:true});
      check(firefox.network.tool_used==="direct_reasoning",
        "Firefox: web comparison not handled by direct reasoning");
      check(firefox.network.page_read_count>=1,
        "Firefox: no real HTTPS page read for the comparison");
      check(firefox.network.source_citation,
        "Firefox: literal source quotations missing");
      check(firefox.network.has_firefox&&firefox.network.has_chrome,
        "Firefox: both browsers were not compared");
      out.v3.firefox={checked:true,web_pages:firefox.network.page_read_count,
        source_citation:true,tool:"direct_reasoning"};
      out.status="NERAL_V3_PR567_CHROME_4_OF_4_PASS";
    }finally{
      out.last_stage=stage;stage="temporary-account-cleanup";
      try{
        const deleted=await api("/api/neral/account","DELETE",{password});
        out.account_deleted=deleted.deleted===true;
      }catch{out.account_deleted=false;}
      if(!out.account_deleted)out.status="BLOCKED_ACCOUNT_CLEANUP";
    }
  }catch(e){
    out.status="NERAL_V3_PR567_CHROME_BLOCKED";
    out.failed_stage=out.last_stage||stage;
    out.error_type=e?.name||"Error";
    out.safe_error_hint=String(e?.message||"unknown").slice(0,155);
  }finally{
    try{ws?.close()}catch{}try{browser?.kill()}catch{}
    if(profile)try{fs.rmSync(profile,{recursive:true,force:true,maxRetries:3,retryDelay:200})}catch{}
    console.log("NERAL_V3_PR567_CHROME_SCALAR "+JSON.stringify(out));
  }
  process.exitCode=out.status==="NERAL_V3_PR567_CHROME_4_OF_4_PASS"&&out.account_deleted?0:1;
}
main().catch(()=>process.exit(1));
