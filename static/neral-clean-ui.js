"use strict";
/* Pure chat chrome: accessible native + menu. Existing controllers still own
 * upload, formulas, web search, speech, model choice, chat persistence & stop. */
(() => {
  const $ = id=>document.getElementById(id);
  const menu=$("quickAddMenu"),formulas=$("formulaLibraryToggle"),file=$("projectFiles");
  const tools=$("extraTools"),model=$("modelDetails");
  if(!menu)return;
  const close=()=>{menu.open=false;};
  formulas?.addEventListener("click",close);
  file?.addEventListener("change",close);
  menu.addEventListener("toggle",()=>{
    if(menu.open){if(tools)tools.open=false;if(model)model.open=false;}
  });
  document.addEventListener("keydown",event=>{
    if(event.key!=="Escape"||!menu.open)return;
    event.preventDefault();close();menu.querySelector?.("summary")?.focus();
  });
  document.addEventListener("pointerdown",event=>{
    if(menu.open&&!menu.contains(event.target))close();
  });
  window.addEventListener?.("neral:load-chat",close);
})();
