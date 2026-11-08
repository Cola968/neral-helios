"use strict";
// Optional local accounts. Existing guest chat stays available without sign-in.
// This module never sends account data to third-party services.
(() => {
  const $ = id => document.getElementById(id);
  const dlg=$("accountDialog"), form=$("accountForm"), list=$("savedChatList");
  let csrf=null, username=null, activeId=null, turns=[], saveQueue=Promise.resolve();
  let authEpoch=0, navigationEpoch=0;
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
    $("accountSettings").hidden=!username;
    $("newSavedChat").title=username?"Neues gespeichertes Gespräch":"Anmelden, um Chats zu speichern";
    if(!username){list.replaceChildren();activeId=null;turns=[];}
  }
  async function refresh(){
    if(!username)return;
    const startedAuth=authEpoch, owner=username;
    const data=await api("/api/neral/account/chats");
    if(startedAuth!==authEpoch || owner!==username)return;
    list.replaceChildren();
    const today=new Date(),yesterday=new Date(today);
    yesterday.setDate(today.getDate()-1);
    const dayKey=date=>date.getFullYear()+"-"+(date.getMonth()+1)+"-"+date.getDate();
    const currentDay=dayKey(today),previousDay=dayKey(yesterday);
    let currentGroup=null;
    for(const chat of data.chats){
      const stamp=new Date(Number(chat.updated)*1000);
      const group=!Number.isFinite(stamp.getTime())?"Früher":
        dayKey(stamp)===currentDay?"Heute":
        dayKey(stamp)===previousDay?"Gestern":"Früher";
      if(group!==currentGroup){
        const heading=document.createElement("div");
        heading.className="sidebar-chat-date";
        heading.textContent=group;
        list.append(heading);
        currentGroup=group;
      }
      const row=document.createElement("div");
      row.className="saved-chat-row";
      const button=document.createElement("button");
      button.type="button";
      button.className="saved-chat-open";
      button.textContent=chat.title;
      button.title=chat.title;
      button.setAttribute("aria-current",activeId===chat.id?"page":"false");
      button.addEventListener("click",()=>openChat(chat.id));
      const remove=document.createElement("button");
      remove.type="button";remove.className="saved-chat-delete";
      remove.textContent="×";remove.title="Chat löschen";
      remove.setAttribute("aria-label",chat.title+" löschen");
      remove.addEventListener("click",async()=>{
        if(!window.confirm("Diesen gespeicherten Chat endgültig löschen?"))return;
        try{
          await mutate("/api/neral/account/chats/"+chat.id,"DELETE");
          if(activeId===chat.id){activeId=null;turns=[];clearChat();}
          await refresh();
        }catch(e){$("error").textContent=e.message;}
      });
      row.append(button,remove);
      list.append(row);
    }
    applyFilter();
  }
  async function openChat(id){
    const startedAuth=authEpoch, owner=username, selected=++navigationEpoch;
    if(!owner)return;
    try{
      const chat=await api("/api/neral/account/chats/"+encodeURIComponent(id));
      if(startedAuth!==authEpoch || owner!==username || selected!==navigationEpoch)return;
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
    const startedAuth=authEpoch, owner=username, selected=++navigationEpoch;
    try{
      const chat=await mutate("/api/neral/account/chats","POST",{title:"Neuer Chat"});
      if(startedAuth!==authEpoch || owner!==username || selected!==navigationEpoch)return;
      activeId=chat.id;turns=[];clearChat();
      await refresh();
      $("prompt").focus();
    }catch(e){$("error").textContent=e.message;}
  }
  async function initialize(){
    const startedAuth=authEpoch;
    try{
      const session=await api("/api/neral/preview/session");
      if(startedAuth!==authEpoch)return;
      csrf=session.csrf;
      const me=await api("/api/neral/account/me");
      if(startedAuth!==authEpoch)return;
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
      authEpoch++;navigationEpoch++;
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
      authEpoch++;navigationEpoch++;
      csrf=data.csrf;
      window.dispatchEvent(new CustomEvent("neral:account-csrf",{detail:{csrf}}));
      username=null;activeId=null;turns=[];clearChat();authUi();
    }catch(e){$("error").textContent=e.message;}
  });
  $("newSavedChat").addEventListener("click",startNew);
  const searchInput=$("chatSearchInput");
  $("searchChats").addEventListener("click",()=>{
    if(!username){dlg.showModal();return;}
    searchInput.hidden=!searchInput.hidden;
    if(!searchInput.hidden)searchInput.focus();
    else{searchInput.value="";applyFilter();}
  });
  function applyFilter(){
    const term=searchInput.value.trim().toLocaleLowerCase("de");
    for(const row of list.querySelectorAll(".saved-chat-row"))
      row.hidden=!row.querySelector(".saved-chat-open").textContent.toLocaleLowerCase("de").includes(term);
    for(const heading of list.querySelectorAll(".sidebar-chat-date")){
      let node=heading.nextElementSibling,hasVisible=false;
      while(node&&!node.classList.contains("sidebar-chat-date")){
        if(node.classList.contains("saved-chat-row")&&!node.hidden)hasVisible=true;
        node=node.nextElementSibling;
      }
      heading.hidden=!hasVisible;
    }
  }
  searchInput.addEventListener("input",applyFilter);
  window.addEventListener("neral:turn-complete",event=>{
    if(!username)return Promise.resolve();
    const user=event.detail?.user, assistant=event.detail?.assistant;
    if(typeof user!=="string"||typeof assistant!=="string")return Promise.resolve();
    const startedAuth=authEpoch, owner=username, selected=navigationEpoch;
    const stillCurrent=()=>startedAuth===authEpoch &&
      owner===username && selected===navigationEpoch;
    // Serialise local writes. Never discard a completed model response merely
    // because another chat save has not finished yet.
    saveQueue=saveQueue.catch(()=>{}).then(async()=>{
      if(!stillCurrent())return;
      const originalId=activeId;
      const next=turns.concat({role:"user",content:user},
                              {role:"assistant",content:assistant}).slice(-80);
      const title=(turns.length
        ? list.querySelector('[aria-current="page"]')?.textContent
        : user.slice(0,75)) || "Neuer Chat";
      try{
        let cid=originalId;
        if(!cid){
          const created=await mutate("/api/neral/account/chats","POST",{title});
          if(!stillCurrent())return;
          cid=created.id;activeId=cid;
        }
        if(!stillCurrent())return;
        await mutate("/api/neral/account/chats/"+cid,"PUT",{title,messages:next});
        if(!stillCurrent())return;
        if(activeId===cid)turns=next;
        await refresh();
      }catch(e){
        if(stillCurrent())$("error").textContent=
          "Antwort angezeigt, aber Speichern fehlgeschlagen: "+e.message;
      }
    });
    return saveQueue;
  });
  window.addEventListener("neral:chat-cleared",()=>{
    navigationEpoch++;
    turns=[];
    // Clearing a conversation must not overwrite or delete a saved chat.
    activeId=null;
    refresh().catch(()=>{});
  });

  // Export is a real, authenticated, same-origin JSON attachment.
  $("exportChats").setAttribute("href","/api/neral/account/export");
  const manageDialog=$("accountManageDialog");
  const manageForm=$("accountManageForm");
  let manageAction=null;

  function openAccountSettings(action){
    if(!username)return;
    manageAction=action;
    $("accountManageTitle").textContent=action==="password"
      ? "Passwort ändern" : "Konto löschen";
    $("accountManageDescription").textContent=action==="password"
      ? "Nach dem Speichern musst du dich erneut anmelden. Andere Sitzungen werden beendet."
      : "Alle gespeicherten Chats und dieses lokale Konto werden endgültig gelöscht.";
    $("accountNewPasswordGroup").hidden=action!=="password";
    $("accountNewPassword").required=action==="password";
    $("accountManageSubmit").textContent=action==="password"
      ? "Passwort ändern" : "Konto endgültig löschen";
    $("accountManageError").textContent="";
    $("accountCurrentPassword").value="";
    $("accountNewPassword").value="";
    manageDialog.showModal();
    $("accountCurrentPassword").focus();
  }

  $("changePassword").addEventListener("click",()=>openAccountSettings("password"));
  $("deleteAccount").addEventListener("click",()=>openAccountSettings("delete"));
  $("accountManageCancel").addEventListener("click",()=>{
    manageDialog.close();
    $("accountCurrentPassword").value="";
    $("accountNewPassword").value="";
  });
  manageForm.addEventListener("submit",async event=>{
    event.preventDefault();
    if(!username || !manageAction)return;
    const current=$("accountCurrentPassword").value;
    const action=manageAction;
    const body=action==="password"
      ? {current_password:current,new_password:$("accountNewPassword").value}
      : {password:current};
    if(action==="delete"&&!window.confirm(
        "Konto und gespeicherte Chats dauerhaft löschen? Dieser Vorgang kann nicht rückgängig gemacht werden."
    ))return;
    $("accountManageSubmit").disabled=true;
    $("accountManageError").textContent="";
    try {
      const endpoint=action==="password"
        ? "/api/neral/account/password" : "/api/neral/account";
      const result=await mutate(endpoint,action==="password"?"POST":"DELETE",body);
      authEpoch++;navigationEpoch++;
      csrf=result.csrf;
      window.dispatchEvent(new CustomEvent("neral:account-csrf",{detail:{csrf}}));
      username=null;activeId=null;turns=[];
      clearChat();authUi();
      $("accountSettings").open=false;
      manageDialog.close();
      $("accountCurrentPassword").value="";
      $("accountNewPassword").value="";
      $("error").textContent=action==="password"
        ? "Passwort geändert. Bitte neu anmelden."
        : "Lokales Konto und gespeicherte Chats gelöscht.";
    } catch(e) {
      $("accountManageError").textContent=e.message||"Änderung fehlgeschlagen.";
    } finally {
      $("accountManageSubmit").disabled=false;
    }
  });
  initialize();
})();
