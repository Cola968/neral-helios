'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const core=require('../static/helios-core.js');
const runtime='qwen2.5-coder:'+'a'.repeat(64);
const base='Qwen/Qwen2.5-Coder-7B-Instruct';
const user={role:'user',content:'Hallo',mode:'chat',imported:false};
const answer={role:'assistant',content:'Hello',mode:'trained-generation',source:'external-model',
  checkpoint:'hf-qwen25-coder-7b',base_model:base,model_family:'qwen2.5-coder',runtime_id:runtime,imported:false,interrupted:false};
const result={...answer,active_checkpoint:answer.checkpoint,text:answer.content,model_generated:true};
const context=messages=>core.context(messages,'Weiter',192,2048,answer.checkpoint,runtime,'qwen2.5-coder');

test('Qwen2.5 streaming completion is accepted with exact external identity',()=>{
  assert.equal(core.generationResult(result),result);
  assert.match(core.sourceLabel(answer),/Qwen2.5.*extern/);
  for(const change of [{source:'native-model'},{base_model:'other'},{model_family:'neral-byte'},
                       {runtime_id:'stale'},{active_checkpoint:'arctus350'}])
    assert.throws(()=>core.generationResult({...result,...change}));
});

test('Qwen2.5 context retains complete pairs and provenance on both messages',()=>{
  const history=context([user,answer]).history;
  assert.equal(history.length,2);
  assert(history.every(m=>m.runtime_id===runtime&&m.model_family==='qwen2.5-coder'));
  for(const change of [{runtime_id:'stale'},{source:'native-model'},{base_model:'other'},
                       {imported:true},{interrupted:true},{model_family:'qwen3.5'}])
    assert.equal(context([user,{...answer,...change}]).history.length,0);
  assert.equal(core.context([user,answer],'Hi',100,4096,'original',null,'neral-byte').history.length,0);
});

test('imports keep external attribution but never trusted runtime identity',()=>{
  const imported=core.parseImport(JSON.stringify({format:'neral-chat-v1',title:'Qwen',messages:[user,answer]}));
  assert.equal(imported.messages[1].base_model,base);
  assert.equal(context(imported.messages).history.length,0);
  assert.match(core.sourceLabel(imported.messages[1]),/nicht verifiziert/);
});

test('Qwen BPE fit uses byte safety limit; server checks real tokenizer budget',()=>{
  assert.equal(context([]).fits,true);
  assert(core.context([],'ä'.repeat(1500),192,2048,answer.checkpoint,runtime,'qwen2.5-coder').fits);
});

// Optional actual browser contract using only API fixtures, never GPU/weights.
// HELIOS_TEST_BROWSER points to an already installed Chromium-family browser.
if(process.env.HELIOS_TEST_BROWSER)test('browser shows external Qwen and streams two isolated chat turns',async()=>{
  const {chromium}=require('playwright'),http=require('node:http'),fs=require('node:fs'),path=require('node:path');
  const root=path.resolve(__dirname,'..');
  const allowed=new Set(['templates/index.html','static/neral.css','static/chat-experience.css',
    'static/neral.js','static/helios-core.js','static/projects.js','static/install.js','static/workspace.js',
    'static/manifest.webmanifest','static/sw.js','static/icon.svg','static/icon-192.png','static/icon-512.png']);
  const server=http.createServer((req,res)=>{
    const file=req.url==='/'?'templates/index.html':req.url==='/manifest.webmanifest'?'static/manifest.webmanifest':req.url==='/sw.js'?'static/sw.js':req.url.slice(1);
    if(!allowed.has(file)){res.writeHead(404);res.end();return;}
    res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':file.endsWith('.png')?'image/png':'text/html');
    res.end(fs.readFileSync(path.join(root,file)));
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try{
    browser=await chromium.launch({headless:true,executablePath:process.env.HELIOS_TEST_BROWSER,args:['--disable-gpu']});
    const page=await browser.newPage();const errors=[],requests=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.route('**/api/**',async route=>{
      const url=new URL(route.request().url());let data={available:false};
      if(url.pathname==='/api/projects')data={projects:[]};
      if(url.pathname==='/api/chats')data={chats:[]};
      if(url.pathname==='/api/info')data={parameters:7615616512,layers:28,heads:28,vocab_size:152064,
        max_seq_len:2048,checkpoint_meta:{},device:'cuda:0',gpu:'RTX 4070 (UI fixture)',
        active_checkpoint:answer.checkpoint,checkpoint_loaded:true,runtime_id:runtime,model_family:answer.model_family,
        checkpoints:[answer.checkpoint,'none'],model_labels:{[answer.checkpoint]:'Qwen2.5-Coder 7B'},
        assistant:{enabled:true,base_model:'qwen3.5:9b'}};
      if(url.pathname==='/api/generate'){
        requests.push(route.request().postDataJSON());
        return route.fulfill({contentType:'application/x-ndjson',body:[{type:'start'},
          {type:'progress',text:'Sichere Teilantwort',generated_tokens:4},
          {type:'done',result:{...result,incomplete:false,finish_reason:'eos',generated_tokens:5,
            token_unit:'BPE-Token',seconds:1,tokens_per_second:5}}].map(x=>JSON.stringify(x)).join('\n')+'\n'});
      }
      await route.fulfill({json:data});
    });
    await page.goto('http://127.0.0.1:'+server.address().port);
    await page.locator('#modelName').filter({hasText:'Qwen2.5-Coder'}).waitFor();
    assert.match(await page.locator('#activity').textContent(),/externe Gewichte/);
    assert(await page.locator('#assistantMode').isHidden());
    for(const prompt of ['Erkläre eine Funktion.','Was bedeutet das?']){
      await page.locator('#prompt').fill(prompt);await page.locator('#send').click();
      await page.waitForFunction(()=>!document.getElementById('log').getAttribute('aria-busy')||document.getElementById('log').getAttribute('aria-busy')==='false');
      await page.locator('#log .message-label').filter({hasText:'Qwen2.5-Coder'}).last().waitFor();
    }
    assert.equal(requests.length,2);
    assert(requests.every(r=>r.runtime_id===runtime));
    assert.equal(requests[1].history.length,2);
    assert(requests[1].history.every(m=>m.runtime_id===runtime&&m.model_family==='qwen2.5-coder'));
    assert.deepEqual(errors,[]);
  }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
});
