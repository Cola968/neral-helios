"use strict";
/* NERAL reference shell: only view controls. Model, accounts, files and speech
 * are owned by their established modules; nothing is simulated here. */
(() => {
  const $=id=>document.getElementById(id);
  const trigger=$("modeDropdownTrigger"),menu=$("modeDropdownMenu");
  const low=$("modeOptionLow"),high=$("modeOptionHigh");
  const radios={
    fast:document.querySelector('input[name="mode"][value="fast"]'),
    think:document.querySelector('input[name="mode"][value="think"]')
  };
  if(!trigger||!menu||!low||!high||!radios.fast||!radios.think)return;
  const current=$("modeDropdownLabel");
  const entries={fast:low,think:high};
  const labels={fast:"Niedrig",think:"Hoch"};

  function refresh(){
    const selected=radios.think.checked?"think":"fast";
    current.textContent=labels[selected];
    for(const [key,option] of Object.entries(entries)){
      option.setAttribute("aria-checked",String(key===selected));
      option.classList.toggle("is-selected",key===selected);
    }
  }
  function close(restore=false){
    menu.hidden=true;trigger.setAttribute("aria-expanded","false");
    if(restore)trigger.focus();
  }
  function open(){
    menu.hidden=false;trigger.setAttribute("aria-expanded","true");
    entries[radios.think.checked?"think":"fast"].focus();
  }
  function choose(value){
    const input=radios[value];if(!input)return;
    input.checked=true;
    input.dispatchEvent(new Event("change",{bubbles:true}));
    refresh();close(true);
  }
  trigger.addEventListener("click",()=>menu.hidden?open():close(true));
  low.addEventListener("click",()=>choose("fast"));
  high.addEventListener("click",()=>choose("think"));
  for(const input of Object.values(radios))input.addEventListener("change",refresh);
  menu.addEventListener("keydown",event=>{
    if(event.key==="Escape"){event.preventDefault();close(true);return;}
    if(["ArrowDown","ArrowUp","Home","End"].includes(event.key)){
      event.preventDefault();
      const active=document.activeElement===high?"think":"fast";
      const next=event.key==="Home"?"fast":event.key==="End"?"think":
        active==="fast"?"think":"fast";
      entries[next].focus();
    }
  });
  document.addEventListener("keydown",event=>{
    if(event.key==="Escape"&&!menu.hidden){event.preventDefault();close(true);}
  });
  document.addEventListener("pointerdown",event=>{
    if(!menu.hidden&&!menu.contains(event.target)&&!trigger.contains(event.target))close();
  });
  refresh();

  // Existing tools menu now lives in the top-right chrome, without cloning
  // controls or attaching duplicate event handlers.
  const tools=$("extraTools"),slot=$("headerToolsSlot");
  if(tools&&slot)slot.append(tools);
  $("showPrivacy")?.addEventListener("click",()=>{
    const details=$("privacyDetails");
    if(!details)return;
    details.open=!details.open;
    if(tools)tools.open=false;
    if(details.open)details.scrollIntoView({block:"nearest",behavior:"auto"});
  });
  // Projects in this local version are explicitly chat attachments, not a
  // fictitious persistent workspace.
  $("projectFilesShortcut")?.addEventListener("click",()=>{
    const input=$("projectFiles");
    if(!input)return;
    if(input.disabled)return;
    input.click();
  });

  // Model replies are inert text. Recognized fenced code gets a local,
  // copyable block. No model HTML, styles or scripts are executed.
  const timeline=$("messages");
  const codeFence=String.fromCharCode(96).repeat(3);
  function enhance(item){
    if(!item.classList?.contains("assistant")||item.dataset.referenceReady)return;
    item.dataset.referenceReady="true";
    const heading=item.querySelector(":scope > strong");
    if(heading && /^NERAL(?:\b|$)/i.test(heading.textContent||"")){
      const full=heading.textContent;
      heading.title=full;heading.textContent="NERAL";
      heading.setAttribute("aria-label",full);
    }
    const body=item.querySelector(":scope > p");
    if(!body)return;
    const original=body.textContent||"";
    if(!original.includes(codeFence))return;
    const pattern=/\x60{3}([A-Za-z0-9_+#.-]{0,24})\r?\n([\s\S]*?)\x60{3}/g;
    const result=document.createElement("div");
    result.className="reference-answer-body";
    let last=0,match,found=false;
    while((match=pattern.exec(original))){
      if(match.index>last){
        const prose=document.createElement("div");
        prose.className="reference-prose";
        prose.textContent=original.slice(last,match.index);
        result.append(prose);
      }
      const codeText=match[2];
      if(/^(?:math|latex|tex|katex)$/i.test(match[1])){
        const formula=document.createElement("div");
        formula.className="reference-math";
        formula.setAttribute("aria-label","Mathematische Formel");
        if(window.NeralMath?.renderBlock(formula,codeText)){
          result.append(formula);
          last=pattern.lastIndex;found=true;
          continue;
        }
      }
      // Any missing KaTeX or unrecognized formula falls back to the
      // original text, never a fabricated fraction or executable markup.
      const block=document.createElement("section");
      block.className="reference-code";
      block.setAttribute("aria-label","Codeblock "+(match[1]||"Code"));
      const header=document.createElement("div");
      header.className="reference-code-head";
      const lang=document.createElement("span");
      lang.textContent=match[1]||"Code";
      const button=document.createElement("button");
      button.type="button";button.className="reference-code-copy";
      button.textContent="Kopieren";
      button.setAttribute("aria-label","Code kopieren");
      button.addEventListener("click",async()=>{
        try{
          await navigator.clipboard.writeText(codeText);
          button.textContent="Kopiert";
        }catch{button.textContent="Kopieren fehlgeschlagen";}
      });
      header.append(lang,button);
      const pre=document.createElement("pre"),code=document.createElement("code");
      // Deliberately avoid an HTML parser. A small optional text-only
      // lexer colors Python-like keywords without treating input as markup.
      const tokens=codeText.split(/(\b(?:def|return|for|in|if|else|elif|while|class|from|import|True|False|None|async|await|raise)\b)/g);
      for(const token of tokens){
        if(/^(?:def|return|for|in|if|else|elif|while|class|from|import|True|False|None|async|await|raise)$/.test(token)){
          const span=document.createElement("span");
          span.className="reference-code-keyword";span.textContent=token;
          code.append(span);
        }else code.append(document.createTextNode(token));
      }
      pre.append(code);block.append(header,pre);result.append(block);
      last=pattern.lastIndex;found=true;
    }
    if(!found)return;
    if(last<original.length){
      const tail=document.createElement("div");tail.className="reference-prose";
      tail.textContent=original.slice(last);result.append(tail);
    }
    body.replaceWith(result);
    // Render prose formulas after code fences are safe inert <pre><code>.
    // KaTeX skips code elements and never evaluates model markup.
    window.NeralMath?.render(result);
  }
  if(timeline){
    timeline.querySelectorAll(".message.assistant").forEach(enhance);
    if(typeof MutationObserver!=="undefined")
      new MutationObserver(records=>{
        for(const record of records)for(const node of record.addedNodes)
          if(node.nodeType===1){
            if(node.matches?.(".message.assistant"))enhance(node);
            node.querySelectorAll?.(".message.assistant").forEach(enhance);
          }
      }).observe(timeline,{childList:true,subtree:false});
  }

  // Dark mode is optional and local; no remote CSS/font/theme dependencies.
  const theme=$("themeToggle");
  if(theme){
    let saved="light";
    try{saved=localStorage.getItem("neral-reference-theme")||"light";}catch{}
    function apply(value){
      const dark=value==="dark";
      document.documentElement.dataset.theme=dark?"dark":"light";
      theme.setAttribute("aria-pressed",String(dark));
      theme.setAttribute("aria-label",dark?"Helles Design einschalten":"Dunkles Design einschalten");
      theme.title=dark?"Helles Design":"Dunkles Design";
    }
    apply(saved);
    theme.addEventListener("click",()=>{
      const next=document.documentElement.dataset.theme==="dark"?"light":"dark";
      apply(next);
      try{localStorage.setItem("neral-reference-theme",next);}catch{}
    });
  }
})();
