"use strict";
// Optional local accounts. Existing guest chat stays available without sign-in.
// This module never sends account data to third-party services.
(() => {
  const $ = id => document.getElementById(id);
  const dlg=$("accountDialog"), form=$("accountForm"), list=$("savedChatList");
  let csrf=null, username=null, activeId=null, turns=[], saving=false;
  const api=async(path,opts={})=>{
    const response=await fetch(path,{credentials:"same-origin",cache:"no-store",...opts});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||"Lokale Anfrage fehlgeschlagen.");
    return data;
  };
  const mutate=(path,method,body)=>{
    if(!csrf)throw new Error("Lokale Sitzung fehlt.");
    return api(path,{method,headers:{"X-Neral-CSRF":csrf,
      ...(body===undefined?{}:{"Content-Type":"application/json"})},
      ...(body===undefined?{}:{body:JSON.stringify(body)})});
  };
  function clearChat(){window.dispatchEvent(new CustomEvent("neral:load-chat",{detail:{messages:[]}}));}
  function authUi(){
    $("accountName").textContent=username||"Gast · lokal";
    $("showLogin").hidden=!!username;
    $("logout").hidden=!username;
    $("newSavedChat").title=username?"Neues gespeichertes Gespräch":"Anmelden, um Chats zu speichern";
    if(!username){list.replaceChildren();activeId=null;turns=[];}
  }
  async function refresh(){
    if(!username)return;
    const data=await api("/api/neral/account/chats");
    list.replaceChildren();
    for(const chat of data.chats){
      const button=document.createElement("button");
      button.type="button";
      button.textContent=chat.title;
      button.title=chat.title;
      button.setAttribute("aria-current",activeId===chat.id?"page":"false");
      button.addEventListener("click",()=>openChat(chat.id));
      list.append(button);
    }
  }
  async function openChat(id){
    try{
      const chat=await api("/api/neral/account/chats/"+encodeURIComponent(id));
      activeId=chat.id;
      turns=chat.messages;
      clearChat();
      window.dispatchEvent(new CustomEvent("neral:load-chat",{detail:{messages:turns}}));
      await refresh();
    }catch(e){$("error").textContent=e.message;}
  }
  async function startNew(){
    if(!username){
      dlg.showModal();
      return;
    }
    try{
      const chat=await mutate("/api/neral/account/chats","POST",{title:"Neuer Chat"});
      activeId=chat.id;turns=[];clearChat();
      await refresh();
      $("prompt").focus();
    }catch(e){$("error").textContent=e.message;}
  }
  async function initialize(){
    try{
      const session=await api("/api/neral/preview/session");
      csrf=session.csrf;
      const me=await api("/api/neral/account/me");
      username=me.authenticated?me.username:null;
      authUi();
      await refresh();
    }catch(e){$("error").textContent=e.message;}
  }
  function accountError(error){$("accountError").textContent=error.message||String(error);}
  async function authenticate(action){
    $("accountError").textContent="";
    const candidate=$("accountUsername").value.trim();
    const password=$("accountPassword").value;
    if(!candidate||password.length<12){accountError(new Error("Mindestens 12 Zeichen für das Passwort."));return;}
    try{
      const result=await mutate("/api/neral/account/"+action,"POST",{username:candidate,password});
      csrf=result.csrf;
      window.dispatchEvent(new CustomEvent("neral:account-csrf",{detail:{csrf}}));
      username=result.username;
      activeId=null;turns=[];clearChat();
      authUi();
      await refresh();
      $("accountPassword").value="";
      dlg.close();
    }catch(e){accountError(e);}
  }
  form.addEventListener("submit",event=>{event.preventDefault();authenticate("login");});
  $("accountRegister").addEventListener("click",()=>authenticate("register"));
  $("showLogin").addEventListener("click",()=>dlg.showModal());
  $("accountCancel").addEventListener("click",()=>{dlg.close();$("accountPassword").value="";});
  $("logout").addEventListener("click",async()=>{
    try{
      const data=await mutate("/api/neral/account/logout","POST");
      csrf=data.csrf;
      window.dispatchEvent(new CustomEvent("neral:account-csrf",{detail:{csrf}}));
      username=null;activeId=null;turns=[];clearChat();authUi();
    }catch(e){$("error").textContent=e.message;}
  });
  $("newSavedChat").addEventListener("click",startNew);
  $("searchChats").addEventListener("click",()=>{
    if(!username){dlg.showModal();return;}
    const term=window.prompt("Gespeicherte Chats durchsuchen")?.trim().toLocaleLowerCase("de");
    if(term===undefined)return;
    for(const button of list.querySelectorAll("button"))
      button.hidden=!button.textContent.toLocaleLowerCase("de").includes(term);
  });
  window.addEventListener("neral:turn-complete",async event=>{
    if(!username)return;
    const user=event.detail?.user, assistant=event.detail?.assistant;
    if(typeof user!=="string"||typeof assistant!=="string")return;
    if(saving)return; // reject overlapping persistence instead of mixing chats.
    saving=true;
    const originalId=activeId;
    const next=turns.concat({role:"user",content:user},{role:"assistant",content:assistant});
    const title=(turns.length?list.querySelector('[aria-current="page"]')?.textContent:user.slice(0,75))||"Neuer Chat";
    try{
      let cid=originalId;
      if(!cid){
        const created=await mutate("/api/neral/account/chats","POST",{title});
        cid=created.id;activeId=cid;
      }
      await mutate("/api/neral/account/chats/"+cid,"PUT",{title,messages:next});
      if(activeId===cid)turns=next;
      await refresh();
    }catch(e){$("error").textContent="Antwort angezeigt, aber Speichern fehlgeschlagen: "+e.message;}
    finally{saving=false;}
  });
  window.addEventListener("neral:chat-cleared",()=>{
    turns=[];
    // Clearing a conversation must not overwrite or delete a saved chat.
    activeId=null;
    refresh().catch(()=>{});
  });
  initialize();
})();
