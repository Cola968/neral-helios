// UI contract tests with explicit API fixtures; never load weights or run inference.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
(async()=>{
 const server=http.createServer((req,res)=>{
  const pathname=new URL(req.url,'http://localhost').pathname;
  const file=pathname==='/'?'templates/index.html':pathname==='/manifest.webmanifest'?'static/manifest.webmanifest':pathname==='/sw.js'?'static/sw.js':pathname.slice(1);
  if(!['templates/index.html','static/neral.css','static/chat-experience.css','static/neral.js','static/helios-core.js','static/projects.js','static/install.js','static/workspace.js','static/manifest.webmanifest','static/sw.js','static/icon.svg','static/icon-192.png','static/icon-512.png'].includes(file)){res.writeHead(404);res.end();return;}
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
   if(url.pathname==='/api/info')data={parameters:20587008,layers:6,heads:8,vocab_size:260,max_seq_len:4096,checkpoint_meta:{},checkpoint_sha256:'test-fixture',device:'cpu',active_checkpoint:'pilot_02',checkpoint_loaded:true,checkpoints:['pilot_02','neral50m'],i1_export_available:false};
   if(url.pathname==='/api/projects')data={projects:[]};
   if(url.pathname==='/api/generate'||url.pathname==='/api/model'){modelCalls++;throw Error('Model requests forbidden in UI tests');}
   await route.fulfill({json:data});
  });
  await page.goto('http://127.0.0.1:'+server.address().port);
  // Independent, genuine Chromium offline-PWA test without mocked API routes.
  // A freshly installed worker must control its first tab before a second reload,
  // otherwise an immediate offline restart can fail despite a cached app shell.
  const offlineContext=await browser.newContext({serviceWorkers:'allow'});
  const offlinePage=await offlineContext.newPage();
  await offlinePage.goto('http://127.0.0.1:'+server.address().port);
  await offlinePage.evaluate(()=>navigator.serviceWorker.ready);
  await offlinePage.waitForFunction(()=>navigator.serviceWorker.controller!==null,null,{timeout:5000});
  await offlinePage.locator('#prompt').fill('Saved before going offline');
  assert(await offlinePage.evaluate(()=>localStorage.getItem(storageKey).includes('Saved before going offline')));
  await offlineContext.setOffline(true);
  await offlinePage.reload({waitUntil:'domcontentloaded'});
  assert.equal(await offlinePage.locator('#prompt').inputValue(),'Saved before going offline');
  assert.equal(await offlinePage.locator('h1').first().isVisible(),true);
  assert.equal(await offlinePage.evaluate(async()=>{
    try{await fetch('/api/info');return 'unexpected-network-response';}catch{return 'offline-api-unavailable';}
  }),'offline-api-unavailable','API must never be cached as an offline response');
  await offlineContext.setOffline(false);
  // Repeated query variants must not grow the shell cache or expose URL variants.
  for(let i=0;i<4;i++)assert.equal((await offlinePage.evaluate(i=>fetch('/?quick='+i).then(r=>r.status),i)),200);
  await offlinePage.waitForTimeout(250);
  const queryCacheKeys=await offlinePage.evaluate(async()=>{
    const cache=await caches.open('neral-shell-v3');
    return (await cache.keys()).map(req=>new URL(req.url)).filter(url=>url.search&&url.pathname==='/').length;
  });
  assert.equal(queryCacheKeys,0,'different query parameters must not create duplicate offline shell entries');
  await offlineContext.close();
  await page.locator('#checkpoint option[value=pilot_02]').waitFor({state:'attached'});
  await page.locator('#modelPicker summary').click();
  assert.equal(await page.locator('#modelOptions button').count(),2);
  assert.equal(await page.locator('#modelOptions button[aria-pressed=true]').count(),1);
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
  const legacy={id:'legacy-import-fixture',title:'Alter importierter Verlauf',imported:true,mode:'chat',draft:'',messages:[
   {role:'user',mode:'chat',content:'Alte Frage'},
   {role:'assistant',mode:'trained-generation',checkpoint:'pilot_02',content:'Ungeprüfter alter Import'}]};
  await page.evaluate(chat=>{const saved=JSON.parse(localStorage.getItem(storageKey)||'[]');saved.push(chat);localStorage.setItem(storageKey,JSON.stringify(saved));localStorage.setItem('neral.helios.active',chat.id);},legacy);
  await page.reload();await page.locator('#log .message-label').filter({hasText:'nicht verifiziert'}).waitFor();
  assert.equal(await page.evaluate(()=>HeliosCore.context(current.messages,'Hi',20,4096,'pilot_02').history.length),0);
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
  assert.deepEqual(errors,[]);assert.equal(modelCalls,0);
  console.log('PASS: light/dark, draft/reload, import/Markdown/XSS, variants/delete, model picker, mobile navigation, projects; zero model requests');
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
