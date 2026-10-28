const {test}=require('node:test');
const assert=require('node:assert/strict');
const core=require('../static/helios-core.js');
test('chat import validates roles and keeps untrusted content literal',()=>{
  const r=core.parseImport(JSON.stringify({format:'neral-chat-v1',title:'<img>',messages:[{role:'assistant',mode:'trained-generation',content:'<script>x</script>'}]}));
  assert.equal(r.title,'<img>');assert.equal(r.imported,true);assert.equal(r.messages[0].content,'<script>x</script>');
  assert.throws(()=>core.parseImport(JSON.stringify({format:'neral-chat-v1',title:'x',messages:[{role:'system',mode:'chat',content:'x'}]})));
  assert.throws(()=>core.parseImport('{'));
  assert.throws(()=>core.parseImport('a'.repeat(2*1024*1024+1)));
});
const pair=(text,checkpoint='pilot_02')=>[{role:'user',mode:'chat',content:text},{role:'assistant',mode:'trained-generation',content:text,checkpoint}];
test('history drops whole pairs and respects byte budget',()=>{
  const r=core.context([...pair('😀'.repeat(20)),...pair('neu')],'Hi',30,100,'pilot_02');
  assert.equal(r.omitted,1);assert.equal(r.history.length,2);assert.equal(r.history[0].role,'user');assert.ok(r.fits);
});
test('history never mixes checkpoints or tools',()=>{
  const rows=[...pair('anderes','neral50m'),{role:'user',mode:'solve',content:'2+2'},{role:'assistant',mode:'solve-tool',content:'4'},...pair('eigen')];
  assert.equal(core.context(rows,'Hi',20,1024,'pilot_02').history.length,2);
});

test('native context excludes interrupted, external and imported answers',()=>{
  for(const extra of [{interrupted:true},{imported:true},{source:'external-model'}]){
    const rows=pair('not native context');Object.assign(rows[1],extra);
    assert.equal(core.context(rows,'Hi',20,1024,'pilot_02').history.length,0);
  }
});

test('unrecognized or external origins never look like the original Neral model',()=>{
  assert.match(core.sourceLabel({mode:'external-model',checkpoint:'Qwen'}),/Externes Modell/);
  assert.match(core.sourceLabel({mode:'trained-generation',checkpoint:'Qwen'}),/nicht verifiziert/);
  assert.match(core.sourceLabel({mode:'error'}),/fehlgeschlagen/);
  assert.match(core.sourceLabel({mode:'solve-tool',source:'tool:calculator'}),/Rechenwerkzeug/);
  assert.match(core.sourceLabel({mode:'trained-generation',checkpoint:'pilot_02'}),/Preview 02/);
});

test('imports preserve honest provenance but never become trusted model context',()=>{
  const input={format:'neral-chat-v1',title:'External',messages:[
    {role:'user',mode:'chat',content:'Hi'},
    {role:'assistant',mode:'external-model',source:'external-model',checkpoint:'Qwen',content:'Example'},
  ]};
  const result=core.parseImport(JSON.stringify(input));
  assert.equal(result.messages[1].source,'external-model');
  assert.equal(result.messages[1].imported,true);
  assert.equal(core.context(result.messages,'Hi',20,1024,'Qwen').history.length,0);
});

test('legacy imported chats retain their untrusted origin after reload and sync',()=>{
  const legacy={id:'legacy',title:'Imported',imported:true,messages:pair('old imported')};
  const migrated=core.normalizeStoredChat(legacy);
  assert(migrated.messages.every(m=>m.imported===true));
  assert.equal(core.context(migrated.messages,'Hi',20,1024,'pilot_02').history.length,0);
  const own=pair('new native').map(m=>({...m,imported:false}));
  const reloaded=core.normalizeStoredChat({...migrated,messages:[...migrated.messages,...own]});
  assert.equal(core.context(reloaded.messages,'Hi',20,1024,'pilot_02').history.length,2);
  assert.match(core.sourceLabel(migrated.messages[1]),/nicht verifiziert/);
});

test('prototype property names are never accepted as known model identities',()=>{
  for(const checkpoint of ['constructor','toString','__proto__']){
    assert.match(core.sourceLabel({mode:'trained-generation',checkpoint}),/nicht verifiziert/);
  }
});

test('transport accepts a final NDJSON line without a newline',async()=>{
  const result={mode:'solve-tool',source:'tool:calculator',model_generated:false,text:'15'};
  const response=new Response(JSON.stringify({type:'done',result}),{headers:{'Content-Type':'application/x-ndjson'}});
  assert.equal((await core.readGenerationResponse(response,()=>{})).text,'15');
});

test('transport rejects missing completion, duplicate completion and false provenance',async()=>{
  const good={mode:'trained-generation',model_generated:true,active_checkpoint:'pilot_02',text:'fixture'};
  for(const text of [JSON.stringify({type:'progress',text:'partial',generated_tokens:1}),
      [1,2].map(()=>JSON.stringify({type:'done',result:good})).join('\n'),
      JSON.stringify({type:'done',result:{...good,model_generated:false}}),
      JSON.stringify({type:'done',result:{...good,source:'external-model'}}),
      JSON.stringify({type:'done',result:{mode:'solve-tool',text:'15',model_generated:false,source:'native-model'}})]){
    const response=new Response(text,{headers:{'Content-Type':'application/x-ndjson'}});
    await assert.rejects(core.readGenerationResponse(response,()=>{}));
  }
});

test('malformed transport is explained without exposing parser internals',async()=>{
  const response=new Response('{private broken payload',{headers:{'Content-Type':'application/x-ndjson'}});
  await assert.rejects(core.readGenerationResponse(response,()=>{}),/Antwortformat/);
});
test('overlong prompt fails without erasing original history',()=>{
  const rows=pair('alt'),before=JSON.stringify(rows);
  assert.equal(core.context(rows,'x'.repeat(200),10,100,'pilot_02').fits,false);
  assert.equal(JSON.stringify(rows),before);
});
test('history uses at most six complete pairs',()=>{
  const r=core.context(Array.from({length:9},()=>pair('a')).flat(),'x',10,4096,'pilot_02');
  assert.equal(r.history.length,12);assert.equal(r.omitted,3);
});
test('incomplete code fences remain readable code',()=>{
  assert.deepEqual(core.blocks('```python\nprint(1)'),[{kind:'code',language:'python',text:'print(1)'}]);
});
function element(tag){return {tag,children:[],textContent:'',append(...xs){this.children.push(...xs);},replaceChildren(){this.children=[];},set innerHTML(v){throw Error('HTML parsing forbidden');}};}
const doc={createElement:element,createTextNode:text=>({tag:'#text',textContent:text})};
const walk=n=>[n,...(n.children||[]).flatMap(walk)];
test('markdown does not turn HTML or unsafe URLs into executable DOM',()=>{
  const n=element('div');core.render(n,'# Titel\n**Fett** und `inline`\n<img src=x onerror=alert(1)>\n[Link](javascript:alert(1))',doc);
  const all=walk(n);assert.ok(all.some(x=>x.tag==='strong'));assert.ok(all.some(x=>x.tag==='code'));
  assert.ok(!all.some(x=>['img','script','a','iframe'].includes(x.tag)));
  assert.ok(all.some(x=>x.textContent.includes('<img')));
});
test('code remains literal and receives a copy control',()=>{
  const n=element('div');core.render(n,'```html\n<script>alert(1)</script>\n```',doc);
  const all=walk(n);assert.ok(all.some(x=>x.tag==='button'));
  assert.ok(all.some(x=>x.tag==='code'&&x.textContent==='<script>alert(1)</script>'));
  assert.ok(!all.some(x=>x.tag==='script'));
});
