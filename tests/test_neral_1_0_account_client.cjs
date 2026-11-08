"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const vm=require("node:vm");
const fs=require("node:fs");
const path=require("node:path");

function deferred(){
  let resolve;
  const promise=new Promise(r=>{resolve=r;});
  return {promise,resolve};
}
function node(){
  return {
    children:[],events:{},attrs:{},className:"",textContent:"",
    value:"",hidden:false,disabled:false,required:false,open:false,
    addEventListener(name,callback){this.events[name]=callback;},
    append(...children){this.children.push(...children);},
    replaceChildren(){this.children=[];},
    setAttribute(name,value){this.attrs[name]=value;},
    getAttribute(name){return this.attrs[name];},
    focus(){},showModal(){this.open=true;},close(){this.open=false;},
    querySelector(selector){
      if(selector===".saved-chat-open")
        return this.children.find(n=>n.className==="saved-chat-open")||null;
      if(selector==='[aria-current="page"]')
        return this.children.flatMap(n=>n.children||[])
          .find(n=>n.attrs["aria-current"]==="page")||null;
      return null;
    },
    querySelectorAll(selector){
      if(selector===".saved-chat-row")
        return this.children.filter(n=>n.className==="saved-chat-row");
      return [];
    }
  };
}
function harness({delayList=false,delayOpen=false,delaySave=false}={}){
  const elements=new Map(),events=new Map(),requests=[];
  const listResult=deferred(),openResult=deferred(),saveResult=deferred();
  const writes=[];
  const get=id=>{
    if(!elements.has(id))elements.set(id,node());
    return elements.get(id);
  };
  const window={
    confirm(){return true;},
    addEventListener(name,fn){events.set(name,fn);},
    dispatchEvent(event){events.get(event.type)?.(event);}
  };
  class CustomEvent{constructor(type,opts={}){this.type=type;this.detail=opts.detail;}}
  const response=data=>({ok:true,json:async()=>data});
  const chat={id:"a".repeat(32),title:"Alice privater Chat",updated:1};
  const fetch=async(url,options={})=>{
    requests.push({url,options});
    if(url.endsWith("/session"))return response({csrf:"initial_csrf"});
    if(url.endsWith("/me"))return response({authenticated:true,username:"alice"});
    if(url==="/api/neral/account/chats" && !options.method){
      return delayList?listResult.promise:response({chats:[chat]});
    }
    if(url==="/api/neral/account/chats/"+chat.id && !options.method){
      return delayOpen?openResult.promise:response({
        id:chat.id,title:chat.title,messages:[{role:"assistant",content:"Alice privat"}]
      });
    }
    if(url==="/api/neral/account/chats/"+chat.id && options.method==="PUT"){
      writes.push(JSON.parse(options.body));
      return delaySave && writes.length===1?saveResult.promise:response({saved:true});
    }
    if(url.endsWith("/logout"))return response({authenticated:false,csrf:"rotated_csrf"});
    throw new Error("Unexpected fetch: "+url);
  };
  const src=fs.readFileSync(path.join(__dirname,"../static/neral-accounts.js"),"utf8");
  vm.runInNewContext(src,{window,document:{getElementById:get,createElement:node},
    fetch,CustomEvent,Error,console},{timeout:2500});
  return {get,events,requests,chat,listResult,openResult,saveResult,writes,
    flush:()=>new Promise(resolve=>setImmediate(resolve))};
}

test("a delayed chat-list result cannot reappear after logout",async()=>{
  const h=harness({delayList:true});
  await h.flush();
  assert.equal(h.get("accountName").textContent,"alice");
  await h.get("logout").events.click();
  assert.equal(h.get("accountName").textContent,"Gast · lokal");
  assert.equal(h.get("savedChatList").children.length,0);
  h.listResult.resolve({ok:true,json:async()=>({chats:[h.chat]})});
  await h.flush();
  assert.equal(h.get("savedChatList").children.length,0,
    "A stale response from Alice may not redraw her chat list after logout");
});

test("a delayed open-chat result cannot reveal private turns after logout",async()=>{
  const h=harness({delayOpen:true});
  await h.flush();
  const button=h.get("savedChatList").children[0].children[0];
  assert.equal(button.className,"saved-chat-open");
  const events=[];
  const prior=h.events.get("neral:load-chat");
  h.events.set("neral:load-chat",event=>{
    events.push(event.detail.messages);
    prior?.(event);
  });
  const pending=button.events.click();
  await h.flush();
  await h.get("logout").events.click();
  h.openResult.resolve({ok:true,json:async()=>({
    id:h.chat.id,title:h.chat.title,
    messages:[{role:"assistant",content:"PRIVATE_ALICE_CONTENT"}]
  })});
  await pending;
  assert.equal(h.get("accountName").textContent,"Gast · lokal");
  assert.ok(!JSON.stringify(events).includes("PRIVATE_ALICE_CONTENT"),
    "A late GET must never resurrect a previous account's chat after logout");
});


test("two completed turns are persisted in order even while the first save waits",async()=>{
  const h=harness({delaySave:true});
  await h.flush();
  const button=h.get("savedChatList").children[0].children[0];
  await button.events.click();
  const complete=h.events.get("neral:turn-complete");
  assert.equal(typeof complete,"function");
  const first=complete({detail:{user:"Erste Frage",assistant:"Erste Antwort"}});
  await h.flush();
  assert.equal(h.writes.length,1);
  const second=complete({detail:{user:"Zweite Frage",assistant:"Zweite Antwort"}});
  await h.flush();
  h.saveResult.resolve({ok:true,json:async()=>({saved:true})});
  await first;
  await second;
  await h.flush();
  assert.equal(h.writes.length,2,"Second answer must not be discarded during an earlier save");
  assert.deepEqual(h.writes[1].messages.slice(-2).map(m=>m.content),
    ["Zweite Frage","Zweite Antwort"]);
});
