// Live installability smoke test. Requires the CPU app on 9686; never sends model requests.
const {chromium}=require('playwright');const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.HELIOS_TEST_BROWSER?{executablePath:process.env.HELIOS_TEST_BROWSER}:{}),args:['--disable-gpu','--disable-software-rasterizer']});
 try{
  const context=await browser.newContext();const page=await context.newPage();let inference=0;
  page.on('request',r=>{if(new URL(r.url()).pathname==='/api/generate')inference++;});
  await page.goto(process.env.HELIOS_TEST_URL||'http://127.0.0.1:9686/');
  await page.locator('#modelOptions button').waitFor({state:'attached'});
  const cdp=await context.newCDPSession(page);const manifest=await cdp.send('Page.getAppManifest');
  assert.deepEqual(manifest.errors,[]);assert.equal(JSON.parse(manifest.data).display,'standalone');
  await page.evaluate(()=>Promise.race([navigator.serviceWorker.ready,new Promise((_,reject)=>setTimeout(()=>reject(Error('No service worker')),10000))]).then(()=>true));
  await page.reload();await page.locator('#prompt').fill('Offline-Entwurf');
  await context.setOffline(true);await page.reload();assert.equal(await page.locator('#prompt').inputValue(),'Offline-Entwurf');
  assert.equal(inference,0);console.log('PASS: live manifest, service worker, offline draft recovery, zero model requests');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
