"use strict";
/* Opt-in Smart UI: no model-generated HTML and no automatic tool calls. */
(function(root){
  function suggestionsFor(input){
    if(typeof input!=="string")return [];
    const v=input.trim();
    if(!v||v.length>4000)return [];
    if(/(?:^|\n)\s*(?:async\s+def\s+|def\s+|class\s+|import\s+|from\s+\w+\s+import\s+|print\s*\()/m.test(v))return ["code"];
    if(v.length<=160&&!/[\r\n]/.test(v)&&/\b(?:heute|aktuell|aktuelle|aktuellen|neueste|neuesten|letzte nachrichten)\b/i.test(v))return ["web"];
    if(/\b(?:erklär(?:e|en)?|versteh(?:e|en)?|wie funktioniert|was bedeutet)\b/i.test(v))return ["steps"];
    return [];
  }
  function splitFences(value){
    const text=typeof value==="string"?value:"";
    const fence=/\x60{3}([A-Za-z0-9_+#.-]{0,22})\r?\n([\s\S]*?)\x60{3}/g;
    const result=[];let start=0,match;
    while((match=fence.exec(text))){
      if(match.index>start)result.push({type:"text",text:text.slice(start,match.index)});
      result.push({type:"code",language:match[1]||"Code",text:match[2]});
      start=fence.lastIndex;
    }
    if(start<text.length||!result.length)result.push({type:"text",text:text.slice(start)});
    return result;
  }
  function splitTables(text){
    if(typeof text!=="string")return [{type:"text",text:""}];
    const lines=text.split("\n"),result=[];let buffer="",i=0;
    const cells=line=>{
      const value=line.trim();
      if(!value.startsWith("|")||!value.endsWith("|"))return null;
      return value.slice(1,-1).split("|").map(x=>x.trim());
    };
    function flush(){if(buffer){result.push({type:"text",text:buffer});buffer="";}}
    while(i<lines.length){
      const header=cells(lines[i]),divider=cells(lines[i+1]||"");
      if(header&&divider&&header.length>=2&&header.length<=8&&
         divider.length===header.length&&divider.every(x=>/^:?-{3,}:?$/.test(x))){
        const rows=[];let j=i+2;
        while(j<lines.length&&rows.length<50){
          const row=cells(lines[j]);if(!row||row.length!==header.length)break;
          rows.push(row);j++;
        }
        if(rows.length){
          flush();result.push({type:"table",headers:header,rows});
          i=j;continue;
        }
      }
      buffer+=lines[i]+(i<lines.length-1?"\n":"");i++;
    }
    flush();return result;
  }
  function parseChartSpec(text){
    let spec;
    try{spec=JSON.parse(text);}catch{return null;}
    if(!spec||typeof spec!=="object"||Array.isArray(spec))return null;
    if(typeof spec.title!=="string"||spec.title.length>80||
       !Array.isArray(spec.labels)||!Array.isArray(spec.values)||
       spec.labels.length<2||spec.labels.length>12||
       spec.labels.length!==spec.values.length)return null;
    const labels=spec.labels,values=spec.values;
    if(labels.some(x=>typeof x!=="string"||x.length<1||x.length>36)||
       values.some(x=>typeof x!=="number"||!Number.isFinite(x)||x<0||x>1e9))
      return null;
    return {title:spec.title,rows:labels.map((label,i)=>({label,value:values[i]}))};
  }
  if(typeof module!=="undefined"&&module.exports)module.exports={suggestionsFor,splitFences,splitTables,parseChartSpec};
  if(typeof document==="undefined")return;
  const $=id=>document.getElementById(id);
  const trigger=$("smartToolsTrigger"),menu=$("smartToolsMenu"),modeButton=$("smartModeTrigger"),
        modeMenu=$("smartModeMenu"),modeLabel=$("smartModeLabel"),prompt=$("prompt"),
        web=$("useWebSearch"),messages=$("messages"),suggestions=$("smartSuggestions");
  if(!trigger||!menu||!modeButton||!modeMenu||!prompt||!messages||!suggestions)return;

  function close(button,panel,focus=false){
    panel.hidden=true;button.setAttribute("aria-expanded","false");
    if(focus)button.focus();
  }
  function open(button,panel){
    if(button!==trigger)close(trigger,menu);
    if(button!==modeButton)close(modeButton,modeMenu);
    panel.hidden=false;button.setAttribute("aria-expanded","true");
    panel.querySelector("button:not([disabled])")?.focus();
  }
  function toggle(button,panel){panel.hidden?open(button,panel):close(button,panel,true);}
  trigger.addEventListener("click",()=>toggle(trigger,menu));
  modeButton.addEventListener("click",()=>toggle(modeButton,modeMenu));
  const modes={low:document.querySelector('input[name="mode"][value="fast"]'),
               high:document.querySelector('input[name="mode"][value="think"]')};
  function setMode(value){
    const input=modes[value];if(!input)return;
    input.checked=true;
    input.dispatchEvent(new Event("change",{bubbles:true}));
    modeLabel.textContent=value==="high"?"Hoch":"Niedrig";
    modeMenu.querySelectorAll("[data-mode]").forEach(option=>
      option.setAttribute("aria-checked",String(option.dataset.mode===value)));
    close(modeButton,modeMenu,true);
  }
  modeMenu.querySelectorAll("[data-mode]").forEach(option=>
    option.addEventListener("click",()=>setMode(option.dataset.mode)));
  modeLabel.textContent=modes.high?.checked?"Hoch":"Niedrig";
  for(const popover of [menu,modeMenu]){
    popover.addEventListener("keydown",event=>{
      if(event.key!=="ArrowDown"&&event.key!=="ArrowUp")return;
      const options=[...popover.querySelectorAll('button:not([disabled])')];
      if(!options.length)return;
      event.preventDefault();
      const index=Math.max(0,options.indexOf(document.activeElement));
      options[(index+(event.key==="ArrowDown"?1:options.length-1))%options.length].focus();
    });
  }
  document.addEventListener("pointerdown",event=>{
    if(!menu.hidden&&!menu.contains(event.target)&&!trigger.contains(event.target))close(trigger,menu);
    if(!modeMenu.hidden&&!modeMenu.contains(event.target)&&!modeButton.contains(event.target))close(modeButton,modeMenu);
  });
  document.addEventListener("keydown",event=>{
    if(event.key!=="Escape")return;
    if(!menu.hidden)close(trigger,menu,true);
    else if(!modeMenu.hidden)close(modeButton,modeMenu,true);
    else closeSidebar();
  });

  const sidebar=$("smartSidebar"),dimmer=$("smartBackdrop"),sidebarToggle=$("smartSidebarToggle");
  function closeSidebar(){
    sidebar?.classList.remove("open");
    if(dimmer)dimmer.hidden=true;
    sidebarToggle?.setAttribute("aria-expanded","false");
  }
  sidebarToggle?.addEventListener("click",()=>{
    const visible=sidebar?.classList.toggle("open")||false;
    if(dimmer)dimmer.hidden=!visible;
    sidebarToggle.setAttribute("aria-expanded",String(visible));
  });
  dimmer?.addEventListener("click",closeSidebar);
  function action(sourceId,targetId,after){
    const source=$(sourceId),target=$(targetId);if(!source||!target)return;
    source.addEventListener("click",()=>{
      close(trigger,menu);target.click();if(after)after();
    });
  }
  action("smartAttach","projectFiles");
  action("smartCodeCheck","codeCheck");
  action("smartVoice","voiceInput");
  action("smartStopVoice","stopVoice");
  action("smartClear","clear",()=>{prompt.value="";renderSuggestions();});
  action("smartNewChat","clear",()=>{prompt.value="";renderSuggestions();closeSidebar();});
  const voiceOutput=$("voiceOutput"),smartOutput=$("smartVoiceOutput");
  if(voiceOutput&&smartOutput){
    smartOutput.checked=voiceOutput.checked;smartOutput.disabled=voiceOutput.disabled;
    smartOutput.addEventListener("change",()=>{voiceOutput.checked=smartOutput.checked;});
  }
  const autoSend=$("voiceAutoSend"),smartAuto=$("smartAutoSend");
  if(autoSend&&smartAuto){
    smartAuto.checked=autoSend.checked;
    smartAuto.addEventListener("change",()=>{autoSend.checked=smartAuto.checked;});
  }
  if($("smartVoice"))$("smartVoice").disabled=$("voiceInput")?.disabled===true;
  if($("smartStopVoice"))$("smartStopVoice").disabled=$("stopVoice")?.disabled===true;
  web?.addEventListener("change",()=>{
    $("pluginStatus").textContent=web.checked?
      "Websuche gewählt. Erst beim Senden geht dein Suchbegriff an einen externen Anbieter.":
      "Websuche aus. Keine Anfrage an Suchanbieter.";
    renderSuggestions();
  });
  $("smartTheme")?.addEventListener("click",()=>{
    document.documentElement.dataset.theme=
      document.documentElement.dataset.theme==="dark"?"light":"dark";
  });

  function suggestion(label,handler){
    const b=document.createElement("button");b.type="button";b.className="smart-suggestion";
    b.textContent=label;b.addEventListener("click",handler);suggestions.append(b);
  }
  function renderSuggestions(){
    suggestions.replaceChildren();
    const kinds=suggestionsFor(prompt.value);
    if(kinds.includes("code"))suggestion("Python-Syntax prüfen",()=>$("codeCheck").click());
    if(kinds.includes("web")&&!web?.checked)suggestion("Websuche verwenden",()=>{
      web.checked=true;web.dispatchEvent(new Event("change",{bubbles:true}));prompt.focus();
    });
    if(kinds.includes("steps"))suggestion("Schrittweise erklären",()=>{
      const v=prompt.value.trim();
      if(v&&!v.startsWith("Erkläre Schritt für Schritt:"))prompt.value="Erkläre Schritt für Schritt: "+v;
      renderSuggestions();prompt.focus();
    });
    suggestions.hidden=!suggestions.childElementCount;
  }
  prompt.addEventListener("input",renderSuggestions);
  prompt.addEventListener("keydown",event=>{
    if(event.key==="Enter"&&!event.shiftKey&&!event.isComposing){
      event.preventDefault();$("chatForm").requestSubmit();
    }
  });
  function copyText(value,button){
    if(!navigator.clipboard?.writeText){button.title="Kopieren nicht verfügbar";return;}
    navigator.clipboard.writeText(value).then(()=>{
      const before=button.textContent;button.textContent="Kopiert";
      window.setTimeout(()=>{button.textContent=before;},1200);
    }).catch(()=>{button.title="Kopieren fehlgeschlagen";});
  }
  function renderTable(part,body){
    const box=document.createElement("div");box.className="smart-table-wrap";
    const table=document.createElement("table");table.className="smart-table";
    const thead=document.createElement("thead"),head=document.createElement("tr");
    const tbody=document.createElement("tbody");
    let rows=part.rows.map(x=>x.slice()),sortColumn=-1,ascending=true;
    function fillRows(){
      tbody.replaceChildren();
      for(const row of rows){
        const tr=document.createElement("tr");
        for(const cell of row){
          const td=document.createElement("td");td.textContent=cell;tr.append(td);
        }
        tbody.append(tr);
      }
    }
    part.headers.forEach((label,index)=>{
      const th=document.createElement("th"),button=document.createElement("button");
      button.type="button";button.textContent=label+" ↕";
      button.title="Spalte sortieren";
      button.addEventListener("click",()=>{
        ascending=sortColumn===index?!ascending:true;sortColumn=index;
        rows.sort((a,b)=>{
          const x=a[index].trim(),y=b[index].trim();
          const nx=Number(x.replace(",",".")),ny=Number(y.replace(",","."));
          const cmp=x!==""&&y!==""&&Number.isFinite(nx)&&Number.isFinite(ny)
            ?nx-ny:x.localeCompare(y,"de",{numeric:true,sensitivity:"base"});
          return ascending?cmp:-cmp;
        });
        fillRows();
      });
      th.append(button);head.append(th);
    });
    thead.append(head);table.append(thead,tbody);fillRows();box.append(table);
    const copy=document.createElement("button");copy.type="button";
    copy.className="smart-table-copy";copy.textContent="Tabelle als TSV kopieren";
    copy.addEventListener("click",()=>copyText(
      [part.headers,...part.rows].map(row=>row.join("\t")).join("\n"),copy));
    box.append(copy);body.append(box);
  }
  function renderChart(spec,body){
    const figure=document.createElement("figure");figure.className="smart-chart";
    const heading=document.createElement("figcaption");heading.textContent=spec.title||"Diagramm";
    figure.append(heading);
    const max=Math.max(...spec.rows.map(x=>x.value),1);
    for(const row of spec.rows){
      const item=document.createElement("div");item.className="smart-chart-row";
      const label=document.createElement("span");label.textContent=row.label;
      const barWrap=document.createElement("div");barWrap.className="smart-chart-track";
      const bar=document.createElement("div");bar.className="smart-chart-bar";
      bar.style.width=(row.value/max*100).toFixed(2)+"%";
      bar.setAttribute("aria-label",row.label+": "+row.value);
      barWrap.append(bar);
      const value=document.createElement("span");value.textContent=String(row.value);
      item.append(label,barWrap,value);figure.append(item);
    }
    const notice=document.createElement("div");notice.className="smart-chart-note";
    notice.textContent="Darstellung aus Modellwerten · nicht unabhängig geprüft";
    figure.append(notice);
    body.append(figure);
  }
  function enhanceAnswer(item){
    if(!item.classList?.contains("assistant")||item.dataset.smartReady==="1")return;
    item.dataset.smartReady="1";
    const body=item.querySelector(".message-body, p");if(!body)return;
    const original=body.textContent||"";
    const parts=splitFences(original).flatMap(part=>
      part.type==="text"?splitTables(part.text):[part]);
    if(parts.some(part=>part.type!=="text")){
      body.replaceChildren();
      for(const part of parts){
        if(part.type==="text"){
          const span=document.createElement("span");span.className="smart-prose";
          span.textContent=part.text;body.append(span);continue;
        }
        if(part.type==="table"){renderTable(part,body);continue;}
        if(part.language.toLowerCase()==="neral-chart"){
          const chart=parseChartSpec(part.text);
          if(chart){renderChart(chart,body);continue;}
        }
        const block=document.createElement("div");block.className="smart-code";
        const bar=document.createElement("div");bar.className="smart-code-bar";
        const lang=document.createElement("span");lang.textContent=part.language;
        const copy=document.createElement("button");
        copy.type="button";copy.className="smart-copy-code";copy.textContent="Kopieren";
        copy.setAttribute("aria-label","Code kopieren");
        copy.addEventListener("click",()=>copyText(part.text,copy));
        bar.append(lang,copy);
        const pre=document.createElement("pre"),code=document.createElement("code");
        code.textContent=part.text;pre.append(code);block.append(bar,pre);body.append(block);
      }
    }
    const buttons=document.createElement("div");buttons.className="smart-answer-actions";
    const copy=document.createElement("button");copy.type="button";copy.textContent="Kopieren";
    copy.addEventListener("click",()=>copyText(original,copy));
    const follow=document.createElement("button");follow.type="button";follow.textContent="Einfacher erklären";
    follow.addEventListener("click",()=>{
      prompt.value="Erkläre deine letzte Antwort einfacher und mit einem konkreten Beispiel.";
      renderSuggestions();prompt.focus();
    });
    buttons.append(copy,follow);item.append(buttons);
  }
  messages.querySelectorAll(".message.assistant").forEach(enhanceAnswer);
  if(typeof MutationObserver!=="undefined")new MutationObserver(records=>{
    for(const record of records)for(const node of record.addedNodes)
      if(node.nodeType===1)enhanceAnswer(node);
  }).observe(messages,{childList:true});
  const fileStatus=$("fileStatus");
  if(fileStatus&&typeof MutationObserver!=="undefined"){
    const update=()=>{fileStatus.hidden=fileStatus.textContent==="Keine Dateien ausgewählt.";};
    update();
    new MutationObserver(update).observe(fileStatus,{childList:true,characterData:true,subtree:true});
  }
  renderSuggestions();
})(typeof globalThis!=="undefined"?globalThis:this);
