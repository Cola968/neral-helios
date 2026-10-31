// True browser smoke test of local NERAL Orbit task creation, persistence, approval.
// No fake task API, no external execution, no private data or browser session.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),http=require('node:http');
const {spawn}=require('node:child_process');
const root=path.resolve(__dirname,'..');

(async()=>{
 const folder=fs.mkdtempSync(path.join(os.tmpdir(),'neral-orbit-test-'));
 const probe=http.createServer();
 await new Promise(r=>probe.listen(0,'127.0.0.1',r));
 const port=probe.address().port;await new Promise(r=>probe.close(r));
 const server=spawn(process.env.HELIOS_TEST_PYTHON||'python',['-u','app.py'],{
  cwd:root,windowsHide:true,
  env:{...process.env,HELIOS_PORT:String(port),HELIOS_STATE_ROOT:path.join(folder,'state'),
       HELIOS_ASSETS_ROOT:path.join(folder,'empty-assets'),HELIOS_PREPARATION_MODE:'1',
       HELIOS_ORBIT_ENABLED:'1',HELIOS_DEVICE:'cpu',CUDA_VISIBLE_DEVICES:''},
  stdio:['ignore','pipe','pipe']});
 let log='',browser;
 for(const s of [server.stdout,server.stderr])s.on('data',b=>{log+=String(b);});
 try{
  const url='http://127.0.0.1:'+port;
  let started=false;
  for(let i=0;i<160;i++){
   try{if((await fetch(url)).ok){started=true;break;}}catch{}
   if(server.exitCode!==null)throw Error('Flask exited: '+log.slice(-2500));
   await new Promise(r=>setTimeout(r,250));
  }
  assert(started,'Flask not ready: '+log.slice(-2500));
  browser=await chromium.launch({headless:true,args:['--disable-gpu']});
  const page=await browser.newPage({viewport:{width:1280,height:880}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);
  await page.locator('#openOrbit').click();
  await page.locator('#taskForm').waitFor();
  assert.match(await page.title(),/NERAL Orbit/);
  assert.match(await page.locator('.notice').textContent(),/keine automatische Ausführung/);
  await page.locator('#kind').selectOption('coding');
  await page.locator('#title').fill('Flask Chat verbessern');
  await page.locator('#goal').fill('Prüfe <img src=x onerror=alert(1)> und erstelle nur geprüfte PRs.');
  await page.locator('#create').click();
  await page.locator('#tasks .task h3').filter({hasText:'Flask Chat verbessern'}).waitFor();
  assert.equal(await page.locator('#tasks img').count(),0);
  assert.match(await page.locator('#tasks').textContent(),/DRAFT/);
  await page.locator('#tasks button').filter({hasText:'Freigabe notieren'}).click();
  await page.locator('#tasks .pill').filter({hasText:'APPROVED_NO_EXECUTOR'}).waitFor();
  assert.match(await page.locator('#tasks').textContent(),/Noch kein Cloud-Worker aktiv/);
  await page.reload();
  await page.locator('#tasks .pill').filter({hasText:'APPROVED_NO_EXECUTOR'}).waitFor();
  await page.locator('#tasks button').filter({hasText:'Abbrechen'}).click();
  await page.locator('#tasks .pill').filter({hasText:'CANCELLED'}).waitFor();
  assert.equal(await page.locator('#tasks button').count(),1);
  await page.locator('#tasks button').filter({hasText:'Archivieren'}).click();
  await page.locator('#tasks .empty').filter({hasText:'Noch keine Aufträge'}).waitFor();
  assert.equal(await page.locator('#tasks .task').count(),0);
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'Orbit mobile overflow');
  assert.deepEqual(errors,[]);
  console.log('PASS Orbit real Flask browser: create, persistence, approval without worker, cancel, archive with history, XSS, mobile');
 }finally{
  if(browser)await browser.close();
  server.kill();
  await new Promise(r=>{if(server.exitCode!==null)return r();server.once('exit',r);setTimeout(r,4000);});
  fs.rmSync(folder,{recursive:true,force:true});
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
