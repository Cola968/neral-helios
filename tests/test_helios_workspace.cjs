// Browser integration against a deterministic fake API; no weights or inference.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),http=require('node:http'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),records=new Map();let partial=false;
(async()=>{
 const server=http.createServer(async(req,res)=>{
  const url=new URL(req.url,'http://localhost');
  const json=(value,status=200)=>{res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(value));};
  if(url.pathname==='/api/info')return json({parameters:50012976,layers:8,heads:12,vocab_size:260,max_seq_len:4096,checkpoint_meta:{},device:'cpu',active_checkpoint:'neral-future',checkpoint_loaded:true,checkpoints:['neral-future'],model_labels:{'neral-future':'Neral · Test version'}});
  if(url.pathname==='/api/chats')return json({chats:[...records.values()]});
  if(url.pathname.startsWith('/api/chats/')){
   let body='';for await(const chunk of req)body+=chunk;
   const data=JSON.parse(body),id=url.pathname.split('/').pop(),revision=records.get(id)?.revision||0;
   if(data.revision!==revision)return json({error:'Conflict'},409);
   records.set(id,{id,revision:revision+1,chat:data.chat,deleted:!!data.deleted});return json({revision:revision+1});
  }
  if(url.pathname==='/api/generate'){
   res.writeHead(200,{'Content-Type':'application/x-ndjson'});
   res.write(JSON.stringify({type:'start'})+'\n');
   res.write(JSON.stringify({type:'progress',text:'Visible partial answer',generated_tokens:24})+'\n');
   return setTimeout(()=>{if(!partial)res.write(JSON.stringify({type:'done',result:{text:'Complete answer',mode:'trained-generation',model_generated:true,generated_tokens:30,seconds:1,tokens_per_second:30,active_checkpoint:'neral-future'}})+'\n');res.end();},900);
  }
  if(url.pathname.startsWith('/api/'))return json({projects:[],available:false});
  const file=url.pathname==='/'?'templates/index.html':url.pathname==='/manifest.webmanifest'?'static/manifest.webmanifest':url.pathname==='/sw.js'?'static/sw.js':url.pathname.slice(1);
  if(!/^(static\/[a-z0-9.-]+|templates\/index.html)$/.test(file)||!fs.existsSync(path.join(root,file))){res.writeHead(404);return res.end();}
  res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':file.endsWith('.webmanifest')?'application/manifest+json':file.endsWith('.png')?'image/png':'text/html');res.end(fs.readFileSync(path.join(root,file)));
 });
 await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
 try{
  browser=await chromium.launch({headless:true,...(process.env.HELIOS_TEST_BROWSER?{executablePath:process.env.HELIOS_TEST_BROWSER}:{}),args:['--disable-gpu','--disable-software-rasterizer']});
  const a=await browser.newContext(),b=await browser.newContext(),p=await a.newPage(),q=await b.newPage(),errors=[];
  for(const page of [p,q])page.on('pageerror',e=>errors.push(e.message));
  const url='http://127.0.0.1:'+server.address().port;
  await p.goto(url);await q.goto(url);
  await p.locator('#modelPicker summary').filter({hasText:'Neral · Test version'}).waitFor();
  await p.locator('#prompt').fill('Shared draft');await p.evaluate(()=>syncChats());
  await q.evaluate(()=>syncChats());await q.locator('#chats button').first().click();
  assert.equal(await q.locator('#prompt').inputValue(),'Shared draft');
  // Both change the same revision; neither copy may disappear.
  await p.locator('#prompt').fill('Desktop edit');await q.locator('#prompt').fill('Phone edit');
  await p.evaluate(()=>syncChats());await q.evaluate(()=>syncChats());
  assert(await q.evaluate(()=>chats.some(c=>c.title.includes('Konfliktkopie')&&c.draft==='Phone edit')));
  assert(await q.evaluate(()=>chats.some(c=>c.draft==='Desktop edit')));
  await q.evaluate(()=>syncChats());
  await p.locator('#chatActions summary').click();await p.locator('#openSettings').click();
  await p.locator('#settingFont').selectOption('large');await p.locator('#settingTheme').selectOption('light');
  await p.locator('#closeSettings').click();await p.reload();
  assert.equal(await p.locator('html').getAttribute('data-font'),'large');
  assert.equal(await p.locator('html').getAttribute('data-theme'),'light');
  await p.keyboard.press('Control+Shift+Space');assert(await p.locator('#prompt').evaluate(e=>e===document.activeElement));
  await p.locator('#prompt').fill('A question');await p.locator('#send').click();
  await p.locator('#log').filter({hasText:'Visible partial answer'}).waitFor();
  await p.locator('#log').filter({hasText:'Complete answer'}).waitFor();
  partial=true;await p.locator('#prompt').fill('Interrupted question');await p.locator('#send').click();
  await p.locator('#status').filter({hasText:'Verbindung unterbrochen'}).waitFor();
  assert(await p.evaluate(()=>current.messages.some(m=>m.interrupted&&m.content==='Visible partial answer')));
  await a.setOffline(true);await p.locator('#prompt').fill('Offline edit');
  assert(await p.evaluate(()=>localStorage.getItem(storageKey).includes('Offline edit')));
  await a.setOffline(false);
  await q.goto(url+'/?quick=solve');assert.equal(await q.evaluate(()=>mode),'solve');
  assert.deepEqual(errors,[]);
  console.log('PASS: two-device sync/conflict preservation, custom model label, settings, quick access, streaming, partial failure, offline draft');
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
