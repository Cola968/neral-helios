"use strict";
/* NERAL scientific typography.
 * KaTeX runs entirely from locally packaged static assets.
 * Model text is always inserted as inert text before this renderer runs.
 * No LaTeX content can enable trust, external resources or HTML injection.
 */
(() => {
  const renderOptions=()=>({
    delimiters:[
      {left:"$$",right:"$$",display:true},
      {left:"\\[",right:"\\]",display:true},
      {left:"\\(",right:"\\)",display:false},
      {left:"$",right:"$",display:false}
    ],
    ignoredTags:["script","noscript","style","textarea","pre","code","option"],
    ignoredClasses:["katex","katex-display","neral-no-math"],
    throwOnError:false,
    errorColor:"#9c4b42",
    trust:false,
    strict:"warn",
    maxExpand:200,
    maxSize:12
  });
  const forbidden=new Set(["PRE","CODE","SCRIPT","STYLE","TEXTAREA","OPTION"]);
  function render(element){
    if(!element || forbidden.has(String(element.tagName||"").toUpperCase()))return false;
    const renderer=typeof window!=="undefined" && window.renderMathInElement;
    if(typeof renderer!=="function")return false;
    try {
      renderer(element,renderOptions());
      return true;
    }catch{
      // Always preserve raw text if the math renderer is unavailable or fails.
      return false;
    }
  }
  function stripDuplicateTable(text){
    // A real Dynamic UI table replaces the SAME markdown table, not prose.
    // Raw text is kept in chat history for copy/export. No content is destroyed.
    if(typeof text!=="string"||text.length>30000)return text;
    const lines=text.split(/\r?\n/),kept=[];
    let i=0,removed=false;
    const cells=line=>/^\s*\|/.test(line)&&
      line.trim().replace(/^\|/,"").replace(/\|$/,"").split("|").length>=2;
    while(i<lines.length){
      if(!removed&&i+2<lines.length&&cells(lines[i])&&
          /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[i+1])&&
          cells(lines[i+2])){
        i+=2;
        while(i<lines.length&&cells(lines[i]))i++;
        removed=true;
        continue;
      }
      kept.push(lines[i]);i++;
    }
    if(!removed)return text;
    return kept.join("\n").replace(/\n{3,}/g,"\n\n").trim();
  }
  if(typeof window!=="undefined")
    window.NeralMath={render,stripDuplicateTable};
})();
