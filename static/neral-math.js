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
  // Math fences are *not* code fences. Only two exactly recognized flattened
  // Kepler forms are repaired; other uncertain flattenings remain untouched.
  // Never turn an arbitrary invalid model equation into an asserted formula.
  function normalizeMathFence(value){
    if(typeof value!=="string" || !value.trim() || value.length>2048)return null;
    let expression=value.trim();
    if(expression.startsWith("$") && expression.endsWith("$"))
      expression=expression.slice(2,-2).trim();
    else if(expression.startsWith("\\[") && expression.endsWith("\\]"))
      expression=expression.slice(2,-2).trim();
    if(!expression || /[<>]/.test(expression) || /[\u0000-\u0008\u000b-\u001f]/.test(expression))
      return null;
    const compact=expression.replace(/\s+/g,"");
    if(compact==="T2a3=4π2G(M+m)")
      return String.raw`\frac{T^2}{a^3}=\frac{4\pi^2}{G(M+m)}`;
    if(compact==="T2a3=4π2GM")
      return String.raw`\frac{T^2}{a^3}=\frac{4\pi^2}{GM}`;
    if(/^T2a3=/i.test(compact))return null;
    // Render already valid LaTeX and clear symbolic formulas. Plain prose or
    // unexplained run-together numbers/letters must never be interpreted.
    if(!/[\\^_=]/.test(expression) || /\\(?:href|url|html|includegraphics|newcommand|def|write18)\b/i.test(expression))
      return null;
    return expression.replace(/π/g,String.raw`\pi `);
  }
  function renderBlock(element,source){
    if(!element || forbidden.has(String(element.tagName||"").toUpperCase()))return false;
    const formula=normalizeMathFence(source);
    const katex=typeof window!=="undefined" && window.katex;
    if(formula===null || !katex || typeof katex.render!=="function")return false;
    try{
      katex.render(formula,element,{
        displayMode:true,output:"htmlAndMathml",throwOnError:true,
        trust:false,strict:"warn",maxExpand:200,maxSize:12
      });
      return true;
    }catch{return false;}
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
  function hideRedundantMarkdownTable(message,source,redraw){
    if(!message||typeof redraw!=="function")return false;
    const shorter=stripDuplicateTable(source);
    if(shorter===source)return false;
    const body=typeof message.querySelector==="function"
      ? message.querySelector(":scope > p") : null;
    if(!body)return false;
    redraw(body,shorter);
    render(body);
    return true;
  }
  if(typeof window!=="undefined")
    window.NeralMath={render,renderBlock,normalizeMathFence,stripDuplicateTable,hideRedundantMarkdownTable};
})();
