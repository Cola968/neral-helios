/* Real headless Chrome/CDP E2E: all inputs are synthetic; delete test account. */
"use strict";
const fs=require("node:fs"),os=require("node:os"),path=require("node:path");
const {spawn}=require("node:child_process"),crypto=require("node:crypto");
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const base="http://127.0.0.1:9690/";
const out={status:"BLOCKED",real_browser:false,slider:[],v1:{ezb:false,energy:false,boxes:false},account_deleted:false,live_sources:0};
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

async function send(prompt,{web=false,kind}){
  stage="v1-"+kind;
  await choose("Luna");
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
  },kind==="ezb"?23000:160000);
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
    const username="e2ev1"+crypto.randomBytes(6).toString("hex");
    password="E2E-V1-"+crypto.randomBytes(18).toString("hex");
    stage="synthetic-account";
    const registered=await api("/api/neral/account/register","POST",{username,password});
    check(registered.authenticated===true,"Account registration failed");
    csrf=registered.csrf;
    await evaluate("window.__e2e.csrf="+JSON.stringify(csrf)+";"+
      "window.dispatchEvent(new CustomEvent('neral:account-csrf',{detail:{csrf:"+JSON.stringify(csrf)+"}}));"+
      "window.__neralLiveNetwork=[];"+
      "(()=>{const live=window.NeralWebLive;if(live&&typeof live.search==='function'){"+
        "const f=live.search.bind(live);live.search=async opts=>{const d=await f(opts);"+
        "window.__neralLiveNetwork.push({type:'search',status:200,"+
        "query:typeof d.query==='string'?d.query:'',external_request:d.external_request===true,"+
        "source_count:Array.isArray(d.results)?d.results.length:0,"+
        "source_https:Array.isArray(d.results)&&d.results.every(s=>String(s.url||'').startsWith('https://'))});"+
        "return d;};}})();"+
      "(()=>{const old=window.fetch.bind(window);window.fetch=async(...args)=>{"+
        "const route=String(args[0]||'');const r=await old(...args);"+
        "if(route.includes('/api/neral/chat/three-tier')){"+
          "r.clone().json().then(d=>{window.__neralLiveNetwork.push("+
          "{type:'chat',status:r.status,"+
          "query:typeof d.query==='string'?d.query:'',"+
          "external_request:d.external_request===true,"+
          "source_count:Array.isArray(d.results)?d.results.length:0,"+
          "source_https:Array.isArray(d.results)&&d.results.every(s=>/^https:\\/\\//.test(s.url||'')),"+
          "verified:d.verified,tool_used:d.tool_used,tier:d.tier,model_generated:d.model_generated,"+
          "scope:d.verification_scope,safe_to_promote:d.safe_to_promote});}).catch(()=>{});"+
        "}return r;};})();true");
    out.account_created=true;
    try{
      const webStarted=Date.now();
      const web=await send("Suche bitte aktuell im Web: Wie hoch ist heute der Leitzins der EZB? Nenne mir deine Quellen.",{web:true,kind:"ezb"});
      out.ezb_elapsed_ms=Date.now()-webStarted;
      check(out.ezb_elapsed_ms<20000,"EZB search exceeds 20-second browser SLA");
      check(/^Werkzeug: Websuche \(\d+ Quellen?\)$/.test(web.status),"Web status not shown");
      const searched=await net("search");
      check(searched.status===200 && searched.external_request===true,"No real public web request");
      check(searched.query==="Leitzins EZB aktuell","Incorrect EZB web query rewriting");
      check(web.source_links.length>0&&web.source_links.every(x=>/^https:\/\//.test(x||"")),"No actual HTTPS source links in browser");
      out.v1.ezb=true;out.live_sources=web.source_links.length;

      const energy=await send("Ein Server verbraucht 340 Watt im Dauerbetrieb. Wie viele Kilowattstunden sind das in 365 Tagen und was kostet das bei 32,4 Cent pro kWh auf den Cent genau?",{kind:"energy"});
      const ee=await net("chat");
      check(ee.status===200 && ee.verified===true && ee.tool_used==="python_sandbox" &&
            ee.safe_to_promote===true && ee.model_generated===false,"Energy not independently Python-verified");
      check(energy.status==="Werkzeug: Python-Sandbox (verifiziert)","Wrong Python status line");
      check(energy.body.includes("2.978,4 kWh")&&energy.body.includes("965,00 €"),"Incorrect energy or exact cost");
      out.v1.energy=true;

      const boxes=await send("Drei Kisten (Rot, Blau, Grün) wiegen zusammen 45 kg. Blau ist doppelt so schwer wie Rot, und Grün ist 5 kg schwerer als Blau. Wie schwer ist jede einzelne Kiste?",{kind:"boxes"});
      const be=await net("chat");
      check(be.status===200 && be.verified===true && be.tool_used==="z3_solver" &&
            be.safe_to_promote===true && be.model_generated===false,"Boxes not independently Z3 verified");
      check(boxes.status==="Werkzeug: Z3-Solver (verifiziert)","Wrong Z3 status line");
      for(const [name,val] of [["Rot","8"],["Blau","16"],["Grün","21"]]){
        check(new RegExp(name+"[^\\d]{0,12}"+val+"\\s*kg").test(boxes.body),"Wrong box result for "+name);
      }
      out.v1.boxes=true;
      out.status="V1_BROWSER_E2E_PASS";
    }finally{
      out.last_stage=stage;stage="synthetic-account-cleanup";
      try{
        const deleted=await api("/api/neral/account","DELETE",{password});
        out.account_deleted=deleted.deleted===true;
      }catch{out.account_deleted=false}
      if(!out.account_deleted)out.status="BLOCKED_ACCOUNT_CLEANUP";
    }
  }catch(e){
    out.status="V1_BROWSER_E2E_BLOCKED";
    out.failed_stage=out.last_stage||stage;
    out.error_type=e?.name||"Error";
    out.safe_error_hint=String(e?.message||"unknown").slice(0,145);
  }finally{
    try{ws?.close()}catch{}try{browser?.kill()}catch{}
    if(profile)try{fs.rmSync(profile,{recursive:true,force:true,maxRetries:3,retryDelay:200})}catch{}
    console.log("NERAL_PR561_LIVE_BROWSER_SCALAR "+JSON.stringify(out));
  }
  process.exitCode=out.status==="V1_BROWSER_E2E_PASS"&&out.account_deleted?0:1;
}
main().catch(()=>process.exit(1));
