'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const core=require('../static/helios-core.js');
const id='neral-arctus350:'+'a'.repeat(64);
const u={role:'user',content:'Frage',mode:'chat',imported:false};
const a={role:'assistant',content:'Antwort',mode:'trained-generation',source:'native-model',checkpoint:'arctus350',model_family:'neral-arctus350',runtime_id:id,imported:false,interrupted:false};
const context=messages=>core.context(messages,'Weiter',128,1024,'arctus350',id,'neral-arctus350');
test('BPE context retains provenance for the exact runtime',()=>{
  const h=context([u,a]).history;
  assert.equal(h.length,2);assert.equal(h[1].runtime_id,id);
  assert.equal(h[1].model_family,'neral-arctus350');
});
test('checkpoint replacements, byte models, tools and imported text never cross into Arctus context',()=>{
  for(const change of [{runtime_id:'old'}, {model_family:'neral-byte'}, {checkpoint:'original'}, {source:'tool:solve'}, {imported:true}, {interrupted:true}, {runtime_id:undefined}])
    assert.equal(context([u,{...a,...change}]).history.length,0);
});
test('Arctus is explicitly experimental after reload; imported claims stay unverified',()=>{
  const chat=core.normalizeStoredChat(JSON.parse(JSON.stringify({id:'test',title:'test',messages:[u,a]})));
  assert.equal(context(chat.messages).history.length,2);
  assert.match(core.sourceLabel(a),/Arctus350 · experimentell/);
  const imported=core.parseImport(JSON.stringify({format:'neral-chat-v1',title:'import',messages:[u,a]}));
  assert.equal(context(imported.messages).history.length,0);
  assert.match(core.sourceLabel(imported.messages[1]),/nicht verifiziert/);
});
