// Browser -> real Flask -> simulated loopback Ollama. No real weights or GPU.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),http=require('node:http'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {spawn}=require('node:child_process');
(async()=>{
 const calls=[];
 const ollama=http.createServer((req,res)=>{
  if(req.method==='GET'&&req.url==='/api/tags'){
   res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({models:[{name:'qwen3.5:9b'}]}));return;
  }
  if(req.method!=='POST'||req.url!=='/api/chat'){res.writeHead(404);res.end();return;}
  let raw='';req.on('data',x=>raw+=x);req.on('end',()=>{
   const body=JSON.parse(raw);calls.push(body);
   const text=body.messages.at(-1).content;
   const respond=()=>{res.writeHead(200,{'Content-Type':'application/json'});
    res.end(JSON.stringify({model:'qwen3.5:9b',done:true,done_reason:'stop',
     message:{role:'assistant',content:'Testantwort: '+text},eval_count:5,eval_duration:1000000000}));};
   if(text==='Bitte warten')setTimeout(respond,800);else respond();
  });
 });
 await new Promise((resolve,reject)=>{ollama.once('error',reject);ollama.listen(11434,'127.0.0.1',resolve);});
 const root=path.resolve(__dirname,'..'),folder=fs.mkdtempSync(path.join(os.tmpdir(),'neral-qwen-ui-'));
 const probe=http.createServer();await new Promise(r=>probe.listen(0,'127.0.0.1',r));
 const port=probe.address().port;await new Promise(r=>probe.close(r));
 const server=spawn(process.env.HELIOS_TEST_PYTHON||'python',['-u','app.py'],{cwd:root,env:{...process.env,
  HELIOS_PORT:String(port),HELIOS_STATE_ROOT:path.join(folder,'state'),HELIOS_ASSETS_ROOT:path.join(folder,'empty'),
  HELIOS_PREPARATION_MODE:'0',HELIOS_ENABLE_ARCTUS350:'1',HELIOS_DEVICE:'cpu',NERAL_ASSISTANT_ENABLED:'1',
  NERAL_ASSISTANT_MODEL:'qwen3.5:9b',CUDA_VISIBLE_DEVICES:''},stdio:['ignore','pipe','pipe']});
 let log='',browser;for(const s of [server.stdout,server.stderr])s.on('data',x=>log+=x);
 try{
  const url='http://127.0.0.1:'+port;
  let ready=false;for(let i=0;i<160;i++){
   try{if((await fetch(url)).ok){ready=true;break;}}catch{}
   if(server.exitCode!==null)throw Error(log);
   await new Promise(r=>setTimeout(r,250));
  }
  assert(ready,log);browser=await chromium.launch({headless:true,args:['--disable-gpu']});
  const page=await browser.newPage({viewport:{width:390,height:844}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));await page.goto(url);
  await page.locator('#assistantMode').waitFor({state:'visible'});
  assert.match(await page.locator('#assistantMode').textContent(),/Qwen/);
  await page.locator('#assistantMode').click();
  assert.equal(await page.locator('#assistantMode').getAttribute('aria-pressed'),'true');
  async function send(value){await page.locator('#prompt').fill(value);await page.locator('#send').click();
   await page.locator('#log .message.assistant .content').filter({hasText:'Testantwort: '+value}).waitFor();}
  await send('Hallo NERAL');
  assert.match(await page.locator('#log .message-label').last().textContent(),/Qwen/);
  assert.equal(await page.locator('#stop').isVisible(),false);
  await send('Noch eine Frage');
  assert.deepEqual(calls.at(-1).messages.slice(-3).map(m=>m.role),['user','assistant','user']);
  const pending=send('Bitte warten');await page.locator('#log .pending').waitFor();
  assert.equal(await page.locator('#stop').isVisible(),false);
  assert.equal(await page.locator('[data-set-mode="chat"]').isDisabled(),true);
  await pending;
  await page.locator('[data-set-mode="chat"]').click();
  await page.locator('#modelPicker summary').click();
  assert.match(await page.locator('[data-checkpoint="arctus350"]').textContent(),/Experimentell/);
  await page.locator('[data-checkpoint="arctus350"]').click();
  await page.locator('#status').filter({hasText:'konfigurieren'}).waitFor();
  await page.locator('#assistantMode').click();await send('Zurück zu Qwen');
  assert.deepEqual(calls.at(-1).messages.slice(-3).map(m=>m.role),['user','assistant','user']);
  await page.waitForFunction(async()=>{
   const response=await fetch('/api/chats');if(!response.ok)return false;
   const data=await response.json();
   return data.chats.some(row=>row.chat?.messages?.some(m=>m.mode==='neral-assistant'&&
    m.base_model==='qwen3.5:9b'&&m.source==='assistant:ollama-qwen'));
  },null,{timeout:15000,polling:500});
  await page.reload();await page.locator('#log .message-label').last().filter({hasText:'Qwen'}).waitFor();
  await new Promise(r=>ollama.close(r));
  await page.locator('#assistantMode').click();await page.locator('#prompt').fill('Offline-Test');await page.locator('#send').click();
  await page.locator('#status').filter({hasText:'Ollama'}).waitFor();
  assert.equal(await page.locator('#prompt').inputValue(),'Offline-Test');
  // The PWA shell must preserve Qwen's label and draft on a fully offline reload.
  await page.evaluate(()=>Promise.race([navigator.serviceWorker.ready,new Promise((_,reject)=>setTimeout(()=>reject(Error('Service worker unavailable')),10000))]));
  await page.locator('#prompt').fill('Offline-Entwurf');
  await page.context().setOffline(true);await page.reload();
  assert.equal(await page.locator('#prompt').inputValue(),'Offline-Entwurf');
  await page.locator('#log .message-label').filter({hasText:'Qwen'}).first().waitFor();
  assert((await page.evaluate(()=>caches.keys())).includes('neral-shell-v4'));
  assert.deepEqual(errors,[]);
  console.log('Flask + browser + simulated loopback Ollama: identity, context, Arctus selection, reload, offline and PWA passed.');
 }finally{if(browser)await browser.close();server.kill();if(ollama.listening)await new Promise(r=>ollama.close(r));fs.rmSync(folder,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1;});
