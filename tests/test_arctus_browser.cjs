// Actual Flask server and model menu. No real checkpoints, no API mocks.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),http=require('node:http');
const {spawn}=require('node:child_process');
(async()=>{
 const root=path.resolve(__dirname,'..'),folder=fs.mkdtempSync(path.join(os.tmpdir(),'neral-arctus-ui-'));
 const probe=http.createServer();await new Promise(r=>probe.listen(0,'127.0.0.1',r));
 const port=probe.address().port;await new Promise(r=>probe.close(r));
 const server=spawn(process.env.HELIOS_TEST_PYTHON||'python',['-u','app.py'],{cwd:root,env:{...process.env,HELIOS_PORT:String(port),HELIOS_STATE_ROOT:path.join(folder,'state'),HELIOS_ASSETS_ROOT:path.join(folder,'empty'),HELIOS_PREPARATION_MODE:'0',HELIOS_ENABLE_ARCTUS350:'1',HELIOS_DEVICE:'cpu',HELIOS_ARCTUS350_CHECKPOINT:'',HELIOS_ARCTUS350_TOKENIZER:'',CUDA_VISIBLE_DEVICES:''},stdio:['ignore','pipe','pipe']});
 let log='',browser;for(const s of [server.stdout,server.stderr])s.on('data',x=>log+=x);
 try{
  const url='http://127.0.0.1:'+port;
  let ready=false;for(let i=0;i<160;i++){
   try{if((await fetch(url)).ok){ready=true;break;}}catch{}
   if(server.exitCode!==null)throw Error(log);
   await new Promise(r=>setTimeout(r,250));
  }
  assert(ready,log);browser=await chromium.launch({headless:true,args:['--disable-gpu']});
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.locator('#connection').filter({hasText:'Werkzeuge bereit'}).waitFor();
  await page.locator('#modelPicker summary').click();
  const arctus=page.locator('[data-checkpoint="arctus350"]');
  assert.match(await arctus.textContent(),/Experimentell/);await arctus.click();
  await page.locator('#status').filter({hasText:'konfigurieren'}).waitFor();
  assert.match(await page.locator('#modelName').textContent(),/Kein Modell/);
  await page.locator('#prompt').fill('7+8');await page.locator('#send').click();
  await page.locator('#log .message.assistant .content').filter({hasText:'15'}).waitFor();
  assert.match(await page.locator('#log .message-label').last().textContent(),/Rechenwerkzeug/);
  await page.locator('#modelPicker summary').click();await page.locator('[data-checkpoint="none"]').click();
  await page.waitForFunction(()=>document.getElementById('checkpoint').value==='none'&&!document.getElementById('checkpoint').disabled);
  await page.reload();await page.locator('#log .message.assistant .content').filter({hasText:'15'}).waitFor();
  assert.deepEqual(errors,[]);console.log('Arctus opt-in menu, missing assets, unload, calculator and reload: passed (no model inference).');
 }finally{if(browser)await browser.close();server.kill();fs.rmSync(folder,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1;});
