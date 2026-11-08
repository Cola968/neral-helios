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

  // Projects in this local version are explicitly chat attachments, not a
  // fictitious persistent workspace.
  $("projectFilesShortcut")?.addEventListener("click",()=>{
    const input=$("projectFiles");
    if(!input)return;
    if(input.disabled)return;
    input.click();
  });

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
