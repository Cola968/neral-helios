'use strict';
const test=require('node:test'), assert=require('node:assert/strict');
const core=require('../static/helios-core.js');
const qwen='qwen3.5:9b',id='neral-arctus350:'+'a'.repeat(64);
const user={role:'user',mode:'assistant',content:'Hallo',imported:false};
const reply={role:'assistant',mode:'neral-assistant',content:'Qwen sagt Hallo',source:'external-model',base_model:qwen,model_family:'qwen3.5',imported:false,interrupted:false};
const history=rows=>core.context(rows,'Weiter',192,4096,qwen,null,'qwen3.5','assistant').history;
test('Qwen assistant context accepts only verified same-model pairs',()=>{
  assert.deepEqual(history([user,reply]),[{role:'user',content:'Hallo'},{role:'assistant',content:'Qwen sagt Hallo'}]);
});
test('native, interrupted, imported or wrong-Qwen turns never leak into Qwen context',()=>{
  for(const change of [{mode:'trained-generation'}, {source:'native-model'}, {base_model:'qwen3.5:4b'},
    {imported:true}, {interrupted:true}, {model_family:'neral-arctus350'}])
    assert.deepEqual(history([user,{...reply,...change}]),[]);
  assert.deepEqual(history([{...user,mode:'chat'},reply]),[]);
  assert.deepEqual(core.context([user,reply],'Weiter',192,1024,'arctus350',id,'neral-arctus350').history,[]);
});
test('Qwen provider provenance is validated and labelled without pretending native weights',()=>{
  const valid={mode:'neral-assistant',text:'Hallo',source:'external-model',base_model:qwen,model_family:'qwen3.5',model_generated:true};
  assert.equal(core.generationResult(valid).source,'external-model');
  assert.match(core.sourceLabel({...reply}),/Qwen3.5/);
  for(const changes of [{source:'native-model'}, {base_model:'arctus350'}, {model_generated:false}])
    assert.throws(()=>core.generationResult({...valid,...changes}));
});
test('Qwen import is marked unverified and not recycled into the active model context',()=>{
  const imported=core.parseImport(JSON.stringify({format:'neral-chat-v1',title:'Qwen',messages:[user,reply]}));
  assert.equal(imported.messages[1].base_model,qwen);
  assert.deepEqual(history(imported.messages),[]);
  assert.match(core.sourceLabel(imported.messages[1]),/nicht verifiziert/);
});
