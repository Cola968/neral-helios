'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const core=require('../static/helios-core.js');
const model='qwen3.5:9b',source='assistant:ollama-qwen';
const user={role:'user',content:'Frage',mode:'assistant',imported:false};
const assistant={role:'assistant',content:'Antwort',mode:'neral-assistant',base_model:model,source,imported:false,interrupted:false};
const history=messages=>core.assistantContext(messages,'Weiter',model);
test('only same Qwen model and genuine completed local turns enter context',()=>{
 const h=history([user,assistant]).history;
 assert.equal(h.length,2);assert.equal(h[1].base_model,model);assert.equal(h[1].source,source);
 for(const change of [{base_model:'qwen3.5:4b'},{source:'native-model'},{source:'tool:calculator'},
   {mode:'trained-generation'},{imported:true},{interrupted:true},{base_model:undefined}])
   assert.equal(history([user,{...assistant,...change}]).history.length,0);
 assert.equal(history([{...user,imported:true},assistant]).history.length,0);
 const smaller={...assistant,base_model:'qwen3.5:4b'};
 assert.equal(core.assistantContext([user,smaller],'Weiter','qwen3.5:4b').history.length,2);
 assert.match(core.sourceLabel(smaller),/Qwen3\.5 4B/);
});
test('Qwen origins are explicit and imported claims are unverified',()=>{
 assert.match(core.sourceLabel(assistant),/Qwen3\.5/);
 assert.match(core.sourceLabel({...assistant,imported:true}),/nicht verifiziert/);
});
test('legacy imports cannot become Qwen context after reload',()=>{
 const saved=core.normalizeStoredChat({id:'t',title:'t',imported:true,messages:[user,assistant].map(({imported,...m})=>m)});
 assert.equal(history(saved.messages).history.length,0);
});
test('Qwen responses must have consistent model and source',async()=>{
 const valid={mode:'neral-assistant',model_generated:true,base_model:model,source,text:'Antwort'};
 async function parse(v){return core.readGenerationResponse(new Response(JSON.stringify(v),{headers:{'Content-Type':'application/json'}}),()=>{});}
 assert.equal((await parse(valid)).base_model,model);
 for(const change of [{source:'native-model'},{source:'tool:calculator'},{base_model:'x'},
  {base_model:undefined},{model_generated:false}])await assert.rejects(parse({...valid,...change}));
});
