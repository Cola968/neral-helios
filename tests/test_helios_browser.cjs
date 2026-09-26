// UI contract tests with explicit API fixtures; never load weights or run inference.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
(async()=>{
 const server=http.createServer((req,res)=>{
  const file=req.url==='/'?'templates/index.html':req.url==='/manifest.webmanifest'?'static/manifest.webmanifest':req.url==='/sw.js'?'static/sw.js':req.url.slice(1);
  if(!['templates/index.html','static/neral.css','static/neral.js','static/helios-core.js','static/projects.js','static/install.js','static/workspace.js','static/manifest.webmanifest','static/sw.js','static/icon.svg','static/icon-192.png','static/icon-512.png'].includes(file)){res.writeHead(404);res.end();return;}
  res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':file.endsWith('.webmanifest')?'application/manifest+json':file.endsWith('.png')?'image/png':file.endsWith('.svg')?'image/svg+xml':'text/html');
  res.end(fs.readFileSync(path.join(root,file)));
 });
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 let browser;
 try{
  browser=await chromium.launch({headless:true,...(process.env.HELIOS_TEST_BROWSER?{executablePath:process.env.HELIOS_TEST_BROWSER}:{}),args:['--disable-gpu','--disable-software-rasterizer']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];let modelCalls=0;
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/api/**',async route=>{
   const url=new URL(route.request().url());let data={available:false};if(url.pathname==='/api/chats')data={chats:[]};
   if(url.pathname==='/api/info')data={parameters:20587008,layers:6,heads:8,vocab_size:260,max_seq_len:4096,checkpoint_meta:{},checkpoint_sha256:'test-fixture',device:'cpu',active_checkpoint:'pilot_02',checkpoint_loaded:true,checkpoints:['pilot_02','neral50m'],models:[{id:'pilot_02',label:'Neral 20M · Preview 02',runtime_family:'legacy-byte-v1',status:'development',selectable:true,active:true},{id:'neral50m',label:'Neral 50M · Sprachpilot',runtime_family:'legacy-byte-v1',status:'development',selectable:true,active:false},{id:'neral-arctus-350',label:'Arctus 350M · Phoenix',runtime_family:'frontier350-bpe-v1',status:'candidate',parameters:349825152,context_tokens:512,selectable:false,active:false,blocked_reason:'Frontier-350-Runtime noch nicht mit Workspace verbunden.'}],i1_export_available:false};
   if(url.pathname==='/api/projects')data={projects:[]};
   if(url.pathname==='/api/generate'||url.pathname==='/api/model'){modelCalls++;throw Error('Model requests forbidden in UI tests');}
   await route.fulfill({json:data});
  });
  await page.goto('http://127.0.0.1:'+server.address().port);
  await page.locator('#checkpoint option[value=pilot_02]').waitFor({state:'attached'});
  await page.locator('#runtimeChipText').waitFor();
  assert.match(await page.locator('#runtimeChipText').textContent(),/pilot_02/);
  assert.equal(await page.locator('#runtimeChip').getAttribute('data-state'),'ready');
  await page.locator('#prompt').fill('Kontextanzeige testen');
  assert.equal(await page.locator('#contextMeter').isVisible(),true);
  assert.match(await page.locator('#contextText').textContent(),/^Kontext \d+ %$/);
  assert.equal(await page.locator('#contextTrack').getAttribute('aria-valuenow')!==null,true);
  await page.locator('#runtimeChip').click();assert.equal(await page.locator('#inspector').isVisible(),true);await page.locator('#closeInspect').click();
  await page.locator('#modelPicker summary').click();
  assert.equal(await page.locator('#modelOptions button').count(),3);
  assert.equal(await page.locator('#modelOptions button[aria-pressed=true]').count(),1);
  assert.equal(await page.locator('#modelOptions button[data-runtime="frontier350-bpe-v1"]').count(),1);
  assert.equal(await page.locator('#modelOptions button[data-runtime="frontier350-bpe-v1"]').isDisabled(),true);
  assert.match(await page.locator('#modelOptions button[data-runtime="frontier350-bpe-v1"]').textContent(),/Arctus 350M/);
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#modelPicker').getAttribute('open'),null);
  assert.equal(await page.locator('.composer-model #checkpoint').count(),1);
  assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  await page.locator('#chatActions summary').click();await page.locator('#theme').click();await page.reload();
  assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
  await page.locator('#prompt').fill('Mein gespeicherter Entwurf');await page.locator('#newChat').click();
  await page.locator('#chats button').filter({hasText:'Neuer Chat'}).first().click();
  assert.equal(await page.locator('#prompt').inputValue(),'Mein gespeicherter Entwurf');
  await page.reload();assert.equal(await page.locator('#prompt').inputValue(),'Mein gespeicherter Entwurf');
  const imported={format:'neral-chat-v1',title:'Sicherheitstest',messages:[{role:'user',mode:'chat',content:'Beispiel'},{role:'assistant',mode:'trained-generation',checkpoint:'pilot_02',content:'# Beispiel\n**Fett** und `inline`\n```python\nprint("Hallo")\n```\n<img src=x onerror=alert(1)>'}]};
  await page.locator('#chatFile').setInputFiles({name:'chat.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(imported))});
  await page.locator('#log .code-block').waitFor();assert.equal(await page.locator('#log img').count(),0);
  assert.equal(await page.locator('#log strong').textContent(),'Fett');
  await page.getByRole('button',{name:'Erneut testen',exact:true}).click();
  assert.equal(await page.locator('#prompt').inputValue(),'Beispiel');
  await page.getByRole('button',{name:'Sicherheitstest',exact:true}).click();
  assert.equal(await page.locator('#log .code-block').count(),1);
  page.once('dialog',d=>d.accept());await page.getByRole('button',{name:'Sicherheitstest löschen',exact:true}).click();
  await page.reload();assert.equal(await page.getByRole('button',{name:'Sicherheitstest',exact:true}).count(),0);
  for(const width of [390,360,1440]){
   await page.setViewportSize({width,height:900});
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'horizontal overflow at '+width);
   const box=await page.locator('#modelPicker summary').boundingBox();assert(box&&box.x>=0&&box.x+box.width<=width,'model picker outside viewport');
   if(width<730){await page.locator('#menu').click();await page.locator('#sidebarBackdrop').click({position:{x:width-10,y:300}});assert.equal(await page.locator('#sidebarBackdrop').isVisible(),false);}
  }
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.locator('#mobileNav').isVisible(),true);
  await page.locator('#mobileSolve').click();assert.match(await page.locator('#prompt').getAttribute('placeholder'),/Rechnung/);
  assert.equal(await page.locator('#mobileSolve').getAttribute('aria-pressed'),'true');
  await page.locator('#mobileProjects').click();await page.locator('#projectWorkspace').waitFor({state:'visible'});
  await page.locator('#mobileChat').click();assert.equal(await page.locator('#projectWorkspace').isVisible(),false);
  await page.locator('#mobileSettings').click();assert.equal(await page.locator('#settingsDialog').isVisible(),true);await page.locator('#closeSettings').click();
  await page.setViewportSize({width:1440,height:900});assert.equal(await page.locator('#mobileNav').isVisible(),false);
  await page.locator('#welcomeProjects').click();await page.locator('#projectWorkspace').waitFor({state:'visible'});
  await page.keyboard.press('Escape');assert.equal(await page.locator('#projectWorkspace').isVisible(),false);
  await page.locator('#chatActions summary').click();await page.locator('#theme').click();
  fs.mkdirSync(path.join(root,'.browser-artifacts'),{recursive:true});
  await page.locator('#modelPicker summary').click();
  await page.screenshot({path:path.join(root,'.browser-artifacts/desktop-dark.png')});
  await page.keyboard.press('Escape');
  await page.locator('#chatActions summary').click();await page.locator('#theme').click();await page.setViewportSize({width:390,height:844});
  await page.locator('#modelPicker summary').click();
  const popup=await page.locator('.model-menu').boundingBox();assert(popup.x>=0&&popup.x+popup.width<=390);
  await page.screenshot({path:path.join(root,'.browser-artifacts/mobile-light.png')});
  await page.keyboard.press('Escape');
  await page.route('**/api/info',route=>route.fulfill({json:{parameters:20587008,layers:6,heads:8,vocab_size:260,max_seq_len:4096,checkpoint_meta:{},device:'cpu',active_checkpoint:'pilot_02',checkpoint_loaded:false,checkpoints:['pilot_02','neral50m'],preparation_mode:true}}));
  await page.reload();await page.locator('#modelOptions button').first().waitFor({state:'attached'});
  assert.equal(await page.locator('#modelOptions button:disabled').count(),2);
  assert.equal(await page.locator('#runtimeChip').getAttribute('data-state'),'tools');
  assert.deepEqual(errors,[]);assert.equal(modelCalls,0);
  console.log('PASS: runtime status, context meter, mobile primary navigation, light/dark, draft/reload, import/Markdown/XSS, variants/delete, model picker, projects; zero model requests');
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
