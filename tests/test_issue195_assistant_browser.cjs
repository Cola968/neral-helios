// Real NERAL Flask + browser + fake local Ollama HTTP. No weights, network or GPU.
'use strict';
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),http=require('node:http');
const {spawn}=require('node:child_process');
const root=path.resolve(__dirname,'..');
(async()=>{
 const folder=fs.mkdtempSync(path.join(os.tmpdir(),'neral-issue195-browser-'));
 const requests=[];
 let ollama=http.createServer((req,res)=>{
  if(req.method!=='POST'||req.url!=='/api/chat'){res.writeHead(404);res.end();return;}
  let body='',size=0;
  req.on('data',chunk=>{size+=chunk.length;if(size>65536){req.destroy();return;}body+=chunk;});
  req.on('end',()=>{
   let payload;try{payload=JSON.parse(body);}catch{res.writeHead(400);res.end();return;}
   requests.push(payload);
   const prompt=payload.messages.at(-1).content;
   const answer={model:payload.model,done:true,message:{role:'assistant',content:'Qwen-Fixture: '+prompt},
                 eval_count:5,eval_duration:1000000000};
   res.writeHead(200,{'Content-Type':'application/json'});
   setTimeout(()=>res.end(JSON.stringify(answer)),prompt==='Warte'?350:0);
  });
 });
 await new Promise((resolve,reject)=>ollama.once('error',reject).listen(11434,'127.0.0.1',resolve));
 const probe=http.createServer();await new Promise(r=>probe.listen(0,'127.0.0.1',r));
 const port=probe.address().port;await new Promise(r=>probe.close(r));
 const server=spawn(process.env.HELIOS_TEST_PYTHON||'python',['-u','app.py'],{
  cwd:root,windowsHide:true,
  env:{...process.env,HELIOS_PORT:String(port),HELIOS_STATE_ROOT:path.join(folder,'state'),
       HELIOS_ASSETS_ROOT:path.join(folder,'empty-assets'),HELIOS_PREPARATION_MODE:'0',
       NERAL_ASSISTANT_ENABLED:'1',NERAL_ASSISTANT_MODEL:'qwen3.5:9b',NERAL_ASSISTANT_THINK:'0',
       HELIOS_DEVICE:'cpu',CUDA_VISIBLE_DEVICES:''},
  stdio:['ignore','pipe','pipe']});
 let serverLog='',browser;
 for(const stream of [server.stdout,server.stderr])stream.on('data',d=>serverLog+=d);
 try{
  const url='http://127.0.0.1:'+port;
  let started=false;
  for(let i=0;i<160;i++){
   try{if((await fetch(url)).ok){started=true;break;}}catch{}
   if(server.exitCode!==null)throw Error('Flask stopped: '+serverLog);
   await new Promise(r=>setTimeout(r,250));
  }
  assert(started,'Flask did not start: '+serverLog);
  browser=await chromium.launch({headless:true,args:['--disable-gpu','--disable-software-rasterizer']});
  const page=await browser.newPage({viewport:{width:390,height:844}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);
  await page.locator('#assistantMode').waitFor({state:'visible'});
  await page.locator('#modeHint').filter({hasText:'Qwen3.5'}).waitFor();
  assert.equal(await page.locator('#modelPicker').isVisible(),false);
  assert.match(await page.locator('#assistantModelLabel').textContent(),/externes Open-Weight-Modell/);
  const ask=async prompt=>{
    await page.locator('#prompt').fill(prompt);
    await page.locator('#send').click();
    await page.locator('#log .message.assistant .content').filter({hasText:'Qwen-Fixture: '+prompt}).last().waitFor();
    assert.match(await page.locator('#log .message-label').last().textContent(),/Qwen3.5 9B/);
  };
  await ask('Frage A');
  assert.equal(requests.length,1);
  assert.deepEqual(requests[0].messages.map(m=>m.role),['system','user']);
  await ask('Frage B');
  assert.deepEqual(requests[1].messages.map(m=>m.role),['system','user','assistant','user']);
  assert.equal(requests[1].messages.at(-2).content,'Qwen-Fixture: Frage A');
  await page.locator('[data-set-mode="chat"]').click();
  await page.locator('#prompt').fill('7+8');await page.locator('#send').click();
  await page.locator('#log .message.assistant .content').filter({hasText:'15'}).last().waitFor();
  assert.match(await page.locator('#log .message-label').last().textContent(),/Rechenwerkzeug/);
  assert.equal(requests.length,2,'native tool response must not invoke Ollama');
  await page.locator('#assistantMode').click();
  await ask('Frage C');
  assert.equal(requests.length,3);
  assert.equal(requests[2].messages.length,6,'only the two verified Qwen pairs are reused');
  assert(requests[2].messages.every(m=>!String(m.content||'').includes('15')),
         'native tool answers must not enter Qwen history');
  await page.reload();
  await page.locator('#log .message-label').filter({hasText:'Qwen3.5 9B'}).last().waitFor();
  assert.equal(await page.locator('#log .message-label').filter({hasText:'Qwen3.5 9B'}).count(),3);
  await page.locator('#assistantMode').click();
  await page.locator('#prompt').fill('Warte');
  await page.locator('#send').click();
  assert.equal(await page.locator('#stop').isVisible(),false,'no fake Qwen stop button');
  await page.locator('#log .message.assistant .content').filter({hasText:'Qwen-Fixture: Warte'}).last().waitFor();
  await page.locator('#assistantDeep').check();
  await page.locator('#modeHint').filter({hasText:'Thinking aktiv'}).waitFor();
  await page.locator('#prompt').fill('Deep-Test');
  await page.locator('#send').click();
  await page.locator('#log .message.assistant .content').filter({hasText:'Qwen-Fixture: Deep-Test'}).last().waitFor();
  const deepRequest=requests.at(-1);
  assert.equal(deepRequest.think,true,'real browser toggle must activate Ollama reasoning');
  assert.equal(deepRequest.options.num_ctx,8192);
  assert.equal(deepRequest.options.num_predict,1024);
  assert.match(await page.locator('#log .message.assistant').last().locator('.meta').textContent(),/Denkmodus/);
  await new Promise(r=>ollama.close(r));ollama=null;
  await page.locator('#prompt').fill('Offline');
  await page.locator('#send').click();
  await page.locator('#status').filter({hasText:'Ollama'}).waitFor();
  assert.equal(await page.locator('#prompt').inputValue(),'Offline','draft remains on offline error');
  assert.deepEqual(errors,[]);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'no mobile horizontal overflow');
  console.log('PASS issue195: browser -> Flask -> fake local Ollama, provenance, separate context, reload, no-stop, 503');
 }finally{
  if(browser)await browser.close();
  server.kill();
  if(ollama)await new Promise(r=>ollama.close(r));
  fs.rmSync(folder,{recursive:true,force:true});
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
