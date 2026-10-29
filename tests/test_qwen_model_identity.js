'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const core=require('../static/helios-core.js');
const model='qwen3.5:9b',source='assistant:ollama-qwen',old='a'.repeat(64),fresh='b'.repeat(64);
const user={role:'user',content:'Frage',mode:'assistant',imported:false};
const answer={role:'assistant',content:'Antwort',mode:'neral-assistant',base_model:model,source,
  model_digest:old,imported:false,interrupted:false};
test('same tag with a new or unknown digest never reuses earlier context',()=>{
 assert.equal(core.assistantContext([user,answer],'Weiter',model,old).history.length,2);
 for(const d of [fresh,undefined,null,'bad']){
  const ctx=core.assistantContext([user,answer],'Weiter',model,d);
  assert.equal(ctx.history.length,0);assert.equal(ctx.incompatible,1);
 }
 const legacy={...answer};delete legacy.model_digest;
 const ctx=core.assistantContext([user,legacy],'Weiter',model,old);
 assert.equal(ctx.history.length,0);assert.equal(ctx.incompatible,1);
});
test('model digest is preserved in JSON imports but never trusted for context',()=>{
 const imported=core.parseImport(JSON.stringify({format:'neral-chat-v1',title:'Test',messages:[user,answer]}));
 assert.equal(imported.messages[1].model_digest,old);
 assert.equal(core.assistantContext(imported.messages,'Weiter',model,old).history.length,0);
});
test('successful Qwen response needs a valid digest',async()=>{
 const valid={mode:'neral-assistant',model_generated:true,base_model:model,source,text:'Antwort',model_digest:old};
 async function parse(v){return core.readGenerationResponse(new Response(JSON.stringify(v),{headers:{'Content-Type':'application/json'}}),()=>{});}
 assert.equal((await parse(valid)).model_digest,old);
 for(const digest of [undefined,'bad','z'.repeat(64)])await assert.rejects(parse({...valid,model_digest:digest}));
});
