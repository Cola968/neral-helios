'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

function harness() {
  const elements = new Map();
  function el(id) {
    if (!elements.has(id)) {
      const handlers = {};
      elements.set(id, {
        id, hidden:id==='agentPanel', disabled:false, value:'',
        textContent:'', handlers, classList:{add(){},remove(){}},
        addEventListener(name, cb, capture=false){
          (handlers[name] ||= []).push({cb,capture});
        },
        async emit(name){
          const event = {preventDefault(){}};
          for (const h of (handlers[name] || []).filter(x => x.capture)) await h.cb(event);
          if (name === 'click' && this.onclick) await this.onclick(event);
          for (const h of (handlers[name] || []).filter(x => !x.capture)) await h.cb(event);
        }
      });
    }
    return elements.get(id);
  }
  const surfaces = ['welcome','log','composerSurface','projectWorkspace'].map(el);
  const document = {
    getElementById:el,
    querySelector(query){ return query==='.composer-wrap' ? el('composerSurface') : null; }
  };
  let steps=0;
  const fetch = async (path, opt) => {
    let data;
    if (path.endsWith('/start')) {
      steps = 0;
      data = {id:'1234567890abcdef12345678', goal:'audit', status:'running', steps_completed:0, progress:0,
              plan:['inventory','summary'],results:{},events:[],can_undo:false};
    } else if (path.endsWith('/next')) {
      steps++;
      data = {id:'1234567890abcdef12345678', goal:'audit',
              status:steps>=2?'completed':'running',steps_completed:steps,
              plan:['inventory','summary'],results:{inventory:{files:2}},events:[],can_undo:false};
    } else if (path.endsWith('/quickfix')) {
      data = {id:'1234567890abcdef12345678',goal:'audit',status:'awaiting_approval',
              steps_completed:2,plan:['inventory','summary'],
              results:{},events:[],can_undo:false,
              proposal:{token:'valid-approval',diff:'--- old\n+++ new\n+def answer():\n'}};
    } else if (path.endsWith('/approve')) {
      data = {id:'1234567890abcdef12345678',goal:'audit',status:'completed',
              steps_completed:2,plan:['inventory','summary'],
              results:{},events:[],can_undo:true};
    } else {
      data = {error:'unexpected'};
    }
    return {ok:!data.error, json:async()=>data};
  };
  vm.runInNewContext(fs.readFileSync('static/agent-mode.js','utf8'),
                     {document,fetch,window:{confirm(){return true;}},console});
  return {el,surfaces};
}

test('agent tab hides chat but restores it on close', async () => {
  const {el,surfaces} = harness();
  await el('agentView').emit('click');
  assert.equal(el('agentPanel').hidden,false);
  assert.equal(surfaces[0].hidden,true);
  await el('agentClose').emit('click');
  assert.equal(el('agentPanel').hidden,true);
  assert.equal(surfaces[0].hidden,false);
});

test('agent runs a bounded multi-step plan and disables finished execution', async () => {
  const {el} = harness();
  el('agentGoal').value='audit';
  el('agentStrategy').value='audit';
  el('agentPath').value='.';
  await el('agentForm').emit('submit');
  assert.equal(el('agentNext').disabled,false);
  await el('agentAll').emit('click');
  assert.match(el('agentState').textContent,/completed/);
  assert.equal(el('agentAll').disabled,true);
  assert.match(el('agentEvidence').textContent,/inventory/);
});


test('switching from agent to projects does not get undone by view restoration', async () => {
  const {el} = harness();
  el('projectShortcut').onclick = () => {
    el('projectWorkspace').hidden = false;
    el('welcome').hidden = true;
  };
  await el('agentView').emit('click');
  await el('projectShortcut').emit('click');
  assert.equal(el('agentPanel').hidden, true);
  assert.equal(el('projectWorkspace').hidden, false);
});


test('quick-fix after completed audit requires visible diff and approval', async () => {
  const {el} = harness();
  el('agentGoal').value = 'audit';
  el('agentStrategy').value = 'audit';
  await el('agentForm').emit('submit');
  await el('agentAll').emit('click');
  assert.match(el('agentState').textContent, /completed/);
  assert.equal(el('agentQuickFix').disabled, false);
  el('agentEditPath').value = 'broken.py';
  el('agentFixKind').value = 'python_colon';
  await el('agentQuickFix').emit('click');
  assert.match(el('agentState').textContent, /awaiting_approval/);
  assert.match(el('agentDiff').textContent, /def answer/);
  assert.equal(el('agentApprove').disabled, false);
  await el('agentApprove').emit('click');
  assert.match(el('agentState').textContent, /completed/);
  assert.equal(el('agentUndo').disabled, false);
});
