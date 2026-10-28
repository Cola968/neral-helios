// Real Flask entrypoint + browser. Empty assets, isolated state, CPU, no inference.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),http=require('node:http');
const {spawn}=require('node:child_process');
const root=path.resolve(__dirname,'..');
(async()=>{
 const folder=fs.mkdtempSync(path.join(os.tmpdir(),'neral-chat-smoke-'));
 const probe=http.createServer();await new Promise(r=>probe.listen(0,'127.0.0.1',r));
 const port=probe.address().port;await new Promise(r=>probe.close(r));
 const server=spawn(process.env.HELIOS_TEST_PYTHON||'python',['-u','app.py'],{cwd:root,windowsHide:true,
  env:{...process.env,HELIOS_PORT:String(port),HELIOS_STATE_ROOT:path.join(folder,'state'),HELIOS_ASSETS_ROOT:path.join(folder,'empty-assets'),HELIOS_PREPARATION_MODE:'1',HELIOS_DEVICE:'cpu',CUDA_VISIBLE_DEVICES:''},stdio:['ignore','pipe','pipe']});
 let serverLog='',browser;for(const stream of [server.stdout,server.stderr])stream.on('data',x=>serverLog+=x);
 try{
  const url='http://127.0.0.1:'+port;
  let started=false;for(let i=0;i<160;i++){
   try{if((await fetch(url)).ok){started=true;break;}}catch{}
   if(server.exitCode!==null)throw Error('Flask stopped: '+serverLog);
   await new Promise(r=>setTimeout(r,250));
  }
  assert(started,'Flask did not start: '+serverLog);
  browser=await chromium.launch({headless:true,...(process.env.HELIOS_TEST_BROWSER?{executablePath:process.env.HELIOS_TEST_BROWSER}:{}),args:['--disable-gpu','--disable-software-rasterizer']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.locator('#connection').filter({hasText:'Werkzeuge bereit'}).waitFor();
  await page.locator('#prompt').fill('😀'.repeat(1025));
  assert.equal(await page.locator('#send').isDisabled(),true);
  await page.locator('[data-set-mode=solve]').click();
  assert.equal(await page.locator('#send').isDisabled(),true,'mode changes retain the byte limit');
  await page.locator('[data-set-mode=chat]').click();
  const send=async(prompt,result)=>{
   await page.locator('#prompt').fill(prompt);
   assert.equal(await page.locator('#send').isEnabled(),true,'tools-only chat must be usable');
   await page.locator('#prompt').press('Enter');
   await page.locator('#log .message.assistant').last().locator('.content').filter({hasText:result}).waitFor();
   assert.match(await page.locator('#log .message-label').last().textContent(),/Rechenwerkzeug/);
  };
  await send('7+8','15');
  await send('Wie viel ist 57 plus 68? Gib nur die Zahl aus.','125');
  await send('What is 57 plus 68? Reply only with the number.','125');
  await send('(7+8)*2','30');
  await page.locator('#log').evaluate(el=>el.scrollTop=0);
  await page.locator('#jumpLatest').waitFor({state:'visible'});
  await page.evaluate(()=>render());
  assert(await page.locator('#log').evaluate(el=>el.scrollTop<2),'background render must preserve reading position');
  await page.locator('#jumpLatest').click();
  assert(await page.locator('#log').evaluate(el=>el.scrollHeight-el.scrollTop-el.clientHeight<2));
  const artifacts=path.join(root,'.browser-artifacts');fs.mkdirSync(artifacts,{recursive:true});
  await page.screenshot({path:path.join(artifacts,'chat-experience-desktop.png')});
  await page.locator('#prompt').fill('Berechne 2+2 und ziehe 1 ab');await page.locator('#send').click();
  await page.locator('#log .message-label').last().filter({hasText:'fehlgeschlagen'}).waitFor();
  assert.equal(await page.locator('#prompt').inputValue(),'Berechne 2+2 und ziehe 1 ab');
  assert.equal(await page.locator('#log .message.user').count(),5,'failed request remains visible');
  await page.reload();await page.locator('#log .message.user').last().filter({hasText:'ziehe 1'}).waitFor();
  assert.equal(await page.locator('#prompt').inputValue(),'Berechne 2+2 und ziehe 1 ab');
  await page.screenshot({path:path.join(artifacts,'chat-experience-error.png')});
  // Imported external content stays explicitly external and literal.
  const imported={format:'neral-chat-v1',title:'Externer Herkunftstest',messages:[
   {role:'user',mode:'chat',content:'Testfrage'},
   {role:'assistant',mode:'external-model',source:'external-model',checkpoint:'External fixture',content:'<img src=x onerror=alert(1)> Externes Beispiel'}]};
  await page.locator('#chatFile').setInputFiles({name:'external.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(imported))});
  await page.locator('#log .message-label').filter({hasText:'Externes Modell'}).waitFor();
  assert.equal(await page.locator('#log img').count(),0);
  for(const width of [360,390,768,1440]){
   await page.setViewportSize({width,height:900});
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'overflow at '+width);
   const input=await page.locator('#prompt').boundingBox();assert(input&&input.x>=0&&input.x+input.width<=width);
   if(width<730){await page.locator('#menu').click();await page.keyboard.press('Escape');assert.equal(await page.locator('#sidebarBackdrop').isVisible(),false);}
  }
  await page.locator('#newChat').click();
  await page.locator('#prompt').fill('Zeile eins');await page.locator('#prompt').press('Shift+Enter');await page.locator('#prompt').press('a');
  assert.match(await page.locator('#prompt').inputValue(),/\na$/);
  await page.locator('#prompt').fill('');
  await page.locator('#chatActions summary').click();await page.locator('#openSettings').click();
  await page.locator('#settingTheme').selectOption('light');await page.locator('#closeSettings').click();
  await page.setViewportSize({width:390,height:480});
  const compactSend=await page.locator('#send').boundingBox();
  assert(compactSend&&compactSend.y>=0&&compactSend.y+compactSend.height<=480,'composer must fit a short mobile viewport');
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(artifacts,'chat-experience-mobile.png')});
  assert.deepEqual(errors,[]);
  console.log('PASS real Flask: DE/EN math, tools-only chat, no partial completion, source labels, failure/draft/reload, external import, XSS, keyboard, 4 viewport widths');
 }finally{
  if(browser)await browser.close();server.kill();
  await new Promise(r=>{if(server.exitCode!==null)return r();server.once('exit',r);setTimeout(r,4000);});
  fs.rmSync(folder,{recursive:true,force:true});
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
