"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");

test("new chat keeps original welcome screen attached instead of losing its DOM node",()=>{
  const elements=new Map();
  let welcomeAttached=true;
  function element(id){
    return {
      id, value:"", textContent:"", hidden:false, disabled:false,
      style:{}, className:"",
      classList:{toggle(){},add(){}},
      handlers:{}, children:[],
      addEventListener(name,fn){this.handlers[name]=fn;},
      focus(){},
      querySelector(sel){return null;},
      append(item){if(!item)throw Error("Cannot append missing welcome node");this.children.push(item);if(item.id==="welcome")welcomeAttached=true;},
      replaceChildren(...items){this.children=[...items];welcomeAttached=items.some(x=>x&&x.id==="welcome");},
    };
  }
  for(const id of ["conversation","notice","prompt","chatForm","accessDialog","newChat",
                   "send","sidebarLed","sidebarStatus","welcome","accessForm",
                   "accessError","accessKey","saveAccess"]){
    elements.set(id,element(id));
  }
  const conversation=elements.get("conversation");
  conversation.children=[elements.get("welcome")];
  const document={
    getElementById(id){if(id==="welcome"&&!welcomeAttached)return null;return elements.get(id)||null;},
    querySelectorAll(){return [];},
    createElement(){throw Error("Unexpected node creation before first message");},
  };
  const context={document,TextEncoder,console,
    fetch:async()=>({ok:true,json:async()=>({available:false})}),
    setTimeout,clearTimeout,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../neral_cloud/static/app.js"),"utf8"),context);
  const button=elements.get("newChat");
  assert.equal(typeof button.handlers.click,"function");
  button.handlers.click();
  assert.equal(welcomeAttached,true);
  assert.equal(conversation.children[0].id,"welcome");
  assert.equal(elements.get("welcome").hidden,false);
});
