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
