"use strict";
/* NERAL Dynamic UI V1.
 * Data comes exclusively from the current local reply or verified local tool.
 * Never execute/render model HTML, fetch a remote URL, or synthesize live data.
 * Pure planning is separately tested; rendering progressively enhances text.
 */
(() => {
  const MAX_REPLY=11000;
  const FENCE=String.fromCharCode(96).repeat(3);
  const VALID_TYPES=new Set(["cards","table","bar","timeline"]);
  const own=(obj,keys)=>obj && typeof obj==="object" && !Array.isArray(obj) &&
    Object.keys(obj).every(k=>keys.includes(k));
  const words=(value,max=200)=>typeof value==="string" && value.trim().length>0 &&
    value.trim().length<=max && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value);
  const safeText=(value,max=200)=>words(value,max)?value.trim():"";
  const toNum=value=>{
    if(typeof value==="number")return Number.isFinite(value)?value:null;
    if(typeof value!=="string" || !/^-?\d{1,9}(?:[.,]\d{1,3})?$/.test(value.trim()))return null;
    const n=Number(value.trim().replace(",","."));
    return Number.isFinite(n)?n:null;
  };
  const fmt=n=>Number.isFinite(n)?new Intl.NumberFormat("de-DE",{maximumFractionDigits:4}).format(n):"Nicht definiert";
  const known=(obj,required,optional=[])=>own(obj,required.concat(optional)) &&
    required.every(k=>Object.prototype.hasOwnProperty.call(obj,k));
  function calculate(a,b,op){
    if(!Number.isFinite(a)||!Number.isFinite(b)||!["-","+","*","/"].includes(op))return null;
    if(op==="/" && b===0)return null;
    const result=op==="+"?a+b:op==="-"?a-b:op==="*"?a*b:a/b;
    return Number.isFinite(result)?result:null;
  }
  function markdownTable(text){
    const lines=text.split(/\r?\n/);
    const cells=line=>{
      if(!/^\s*\|/.test(line))return null;
      const raw=line.trim().replace(/^\|/,"").replace(/\|$/,"").split("|").map(x=>x.trim());
      return raw.length>=2 && raw.length<=6 && raw.every(x=>x.length<=160)?raw:null;
    };
    for(let i=0;i+2<lines.length;i++){
      const headers=cells(lines[i]),sep=cells(lines[i+1]);
      if(!headers||!sep||sep.length!==headers.length||
         !sep.every(x=>/^:?-{3,}:?$/.test(x)))continue;
      if(!headers.every(x=>words(x,70)))continue;
      const rows=[];
      for(let j=i+2;j<lines.length&&rows.length<20;j++){
        const row=cells(lines[j]);
        if(!row||row.length!==headers.length)break;
        rows.push(row);
      }
      if(rows.length>=2)return {headers,rows};
    }
    return null;
  }
  function comparisonCards(prompt,text){
    if(!/\b(vergleich|vergleiche|unterschied|compare|optionen|alternativen)\b/i.test(prompt))
      return null;
    const items=[];
    for(const line of text.split(/\r?\n/)){
      const match=line.match(/^\s*(?:[-*]|\d+[.)])\s+\*\*(.{1,70}?)\*\*\s*:?\s*(.{3,250})\s*$/);
      if(!match)continue;
      const title=match[1].replace(/:\s*$/,"").trim(),detail=match[2].trim();
      if(words(title,60)&&words(detail,250)&&!items.some(x=>x.title===title))
        items.push({title,detail});
      if(items.length===6)break;
    }
    return items.length>=2?items:null;
  }
  function mathFromRequest(prompt,result){
    const verified=result && result.model_generated===false &&
      result.tool_used==="neral_math_exact" && result.exact===true;
    if(!verified || typeof prompt!=="string")return null;
    const m=prompt.trim().match(/^(?:bitte\s+)?(?:berechne|calculate|rechne)\s*:?\s*(-?\d+(?:[.,]\d+)?)\s*([+\-*/])\s*(-?\d+(?:[.,]\d+)?)\s*\??$/i);
    if(!m)return null;
    const a=toNum(m[1]),b=toNum(m[3]);
    if(a===null||b===null||calculate(a,b,m[2])===null)return null;
    return {kind:"calculator",title:"Interaktiver Rechner",a,b,op:m[2],provenance:"verified_local"};
  }
  function validatedBlock(obj){
    if(!own(obj,["version","type","title","headers","rows","points","items"]) ||
       obj.version!==1 || !VALID_TYPES.has(obj.type) ||
       (obj.title!==undefined&&!words(obj.title,100)))return null;
    const title=safeText(obj.title,100);
    if(obj.type==="cards"){
      if(!known(obj,["version","type","items"],["title"]) || !Array.isArray(obj.items) ||
         obj.items.length<2 || obj.items.length>6 || !obj.items.every(x=>
           known(x,["title","detail"])&&words(x.title,60)&&words(x.detail,450)))return null;
      return {kind:"cards",title:title||"Optionen",items:obj.items.map(x=>({title:x.title,detail:x.detail}))};
    }
    if(obj.type==="timeline"){
      if(!known(obj,["version","type","items"],["title"])||!Array.isArray(obj.items)||
        obj.items.length<2||obj.items.length>12||!obj.items.every(x=>
          known(x,["label","detail"])&&words(x.label,80)&&words(x.detail,450)))return null;
      return {kind:"timeline",title:title||"Zeitlicher Verlauf",items:obj.items.map(x=>({label:x.label,detail:x.detail}))};
    }
    if(obj.type==="table"){
      if(!known(obj,["version","type","headers","rows"],["title"])||
         !Array.isArray(obj.headers)||obj.headers.length<2||obj.headers.length>6||
         !obj.headers.every(x=>words(x,70))||!Array.isArray(obj.rows)||
         obj.rows.length<2||obj.rows.length>20||!obj.rows.every(row=>
           Array.isArray(row)&&row.length===obj.headers.length&&row.every(x=>words(x,160))))return null;
      return {kind:"table",title:title||"Datentabelle",headers:obj.headers,rows:obj.rows};
    }
    if(obj.type==="bar"){
      if(!known(obj,["version","type","points"],["title"])||!Array.isArray(obj.points)||
         obj.points.length<2||obj.points.length>12||!obj.points.every(x=>
           known(x,["label","value"])&&words(x.label,70)&&typeof x.value==="number"&&
           Number.isFinite(x.value)&&Math.abs(x.value)<=1e9))return null;
      return {kind:"bar",title:title||"Balkendiagramm",points:obj.points.map(x=>({...x}))};
    }
    return null;
  }
  function plan({prompt="",text="",result={}}={}){
    if(typeof prompt!=="string"||typeof text!=="string"||
       text.length<1||text.length>MAX_REPLY)return null;
    const blockStart=text.indexOf(FENCE+"neral-ui\n");
    if(blockStart!==-1){
      const start=blockStart+FENCE.length+"neral-ui\n".length;
      const end=text.indexOf(FENCE,start);
      if(end<0||end-start>6000)return null;
      try {
        const value=JSON.parse(text.slice(start,end));
        const parsed=validatedBlock(value);
        return parsed?{...parsed,provenance:"model_unverified"}:null;
      }catch{return null;}
    }
    const math=mathFromRequest(prompt,result);
    if(math)return math;
    const table=markdownTable(text);
    if(table){
      const wantsChart=/\b(balkendiagramm|säulendiagramm|diagramm|balken|chart|grafik|visualisiere|plot)\b/i.test(prompt);
      if(wantsChart && table.rows.length<=12){
        const column=table.headers.length-1;
        const numbers=table.rows.map(row=>toNum(row[column]));
        if(numbers.every(n=>n!==null && Math.abs(n)<=1e9)){
          return {kind:"bar",title:table.headers[column],points:
            table.rows.map((row,i)=>({label:row[0],value:numbers[i]})),provenance:"model_unverified"};
        }
      }
      if(wantsChart)return null; // No numeric evidence: do not imply data exists.
      return {kind:"table",title:"Interaktive Tabelle",...table,provenance:"model_unverified"};
    }
    const cards=comparisonCards(prompt,text);
    if(cards)return {kind:"cards",title:"Optionen vergleichen",items:cards,provenance:"model_unverified"};
    return null;
  }
  const el=(tag,cls="",text)=>{
    const node=document.createElement(tag);
    if(cls)node.className=cls;
    if(text!==undefined)node.textContent=String(text);
    return node;
  };
  function tableView(host,spec){
    const toolbar=el("div","neral-dui-toolbar");
    const search=el("input","neral-dui-search");
    search.type="search";search.placeholder="Tabelle durchsuchen";
    search.setAttribute("aria-label","Tabelle durchsuchen");
    toolbar.append(search);
    const scroll=el("div","neral-dui-scroll");
    const table=el("table","neral-dui-table");
    const thead=el("thead"),tr=el("tr"),body=el("tbody");
    let sorting=-1,ascending=true;
    const renderRows=()=>{
      const query=String(search.value||"").toLocaleLowerCase("de-DE");
      const rows=spec.rows.filter(row=>row.some(v=>v.toLocaleLowerCase("de-DE").includes(query))).slice();
      if(sorting>=0)rows.sort((a,b)=>{
        const first=toNum(a[sorting]),second=toNum(b[sorting]);
        const delta=first!==null&&second!==null?first-second:
          a[sorting].localeCompare(b[sorting],"de",{numeric:true});
        return ascending?delta:-delta;
      });
      const nodes=rows.map(row=>{
        const tr=el("tr");
        for(const cell of row)tr.append(el("td","",cell));
        return tr;
      });
      if(!nodes.length){
        const empty=el("tr");const cell=el("td","","Keine passenden Zeilen");
        cell.colSpan=spec.headers.length;empty.append(cell);nodes.push(empty);
      }
      body.replaceChildren(...nodes);
    };
    spec.headers.forEach((header,index)=>{
      const th=el("th");
      const btn=el("button","neral-dui-sort",header+" ↕");
      btn.type="button";
      btn.addEventListener("click",()=>{
        if(sorting===index)ascending=!ascending;
        else {sorting=index;ascending=true;}
        renderRows();
      });
      th.append(btn);tr.append(th);
    });
    thead.append(tr);table.append(thead,body);scroll.append(table);
    search.addEventListener("input",renderRows);
    host.append(toolbar,scroll);renderRows();
  }
  function barView(host,spec){
    const max=Math.max(1,...spec.points.map(x=>Math.abs(x.value)));
    const list=el("div","neral-dui-bars");
    for(const point of spec.points){
      const item=el("div","neral-dui-bar-row");
      const label=el("span","neral-dui-bar-label",point.label);
      const track=el("div","neral-dui-track");
      const fill=el("div","neral-dui-fill");
      if(fill.style)fill.style.width=(Math.max(0,Math.min(100,100*Math.abs(point.value)/max)))+"%";
      track.append(fill);
      const number=el("span","neral-dui-bar-value",fmt(point.value));
      item.append(label,track,number);list.append(item);
    }
    host.append(list);
  }
  function cardsView(host,spec){
    const tabs=el("div","neral-dui-cards"),detail=el("p","neral-dui-card-detail");
    const controls=spec.items.map(item=>{
      const btn=el("button","neral-dui-card-button",item.title);
      btn.type="button";
      btn.addEventListener("click",()=>select(item,btn));
      tabs.append(btn);return btn;
    });
    function select(item,selected){
      controls.forEach(b=>b.setAttribute("aria-pressed",b===selected?"true":"false"));
      detail.textContent=item.detail;
    }
    host.append(tabs,detail);select(spec.items[0],controls[0]);
  }
  function timelineView(host,spec){
    const list=el("div","neral-dui-timeline");
    spec.items.forEach((item,index)=>{
      const details=el("details","neral-dui-event");
      if(index===0)details.open=true;
      const summary=el("summary","",item.label);
      const body=el("p","",item.detail);
      details.append(summary,body);list.append(details);
    });
    host.append(list);
  }
  function calculatorView(host,spec){
    const inputs=el("div","neral-dui-calculator");
    const first=el("input"),second=el("input"),selector=el("select");
    first.type="number";second.type="number";first.step="any";second.step="any";
    first.value=String(spec.a);second.value=String(spec.b);
    first.setAttribute("aria-label","Erste Zahl");
    second.setAttribute("aria-label","Zweite Zahl");
    selector.setAttribute("aria-label","Rechenoperation");
    ["+","-","*","/"].forEach(op=>{
      const opt=el("option","",op);opt.value=op;selector.append(opt);
    });
    selector.value=spec.op;
    const output=el("output","neral-dui-result");
    function recalc(){
      const a=toNum(first.value),b=toNum(second.value);
      const result=a===null||b===null?null:calculate(a,b,selector.value);
      output.textContent=result===null?"Nicht definiert":fmt(result);
    }
    first.addEventListener("input",recalc);
    second.addEventListener("input",recalc);
    selector.addEventListener("change",recalc);
    inputs.append(first,selector,second,el("span","neral-dui-equals","="),output);
    host.append(inputs);recalc();
  }
  function mount(message,spec){
    if(!message||!spec||!["table","bar","cards","calculator","timeline"].includes(spec.kind))
      return false;
    try {
      const widget=el("section","neral-dynamic-widget");
      widget.setAttribute("aria-label","Interaktive Darstellung");
      widget.append(el("h3","neral-dui-title",spec.title));
      const content=el("div","neral-dui-content");
      if(spec.kind==="table")tableView(content,spec);
      if(spec.kind==="bar")barView(content,spec);
      if(spec.kind==="cards")cardsView(content,spec);
      if(spec.kind==="timeline")timelineView(content,spec);
      if(spec.kind==="calculator")calculatorView(content,spec);
      const source=spec.provenance==="verified_local"
        ?"Rechnung lokal geprüft · Werte können geändert werden"
        :"Aus NERALs Antwort erstellt · Angaben ungeprüft";
      widget.append(content,el("p","neral-dui-provenance",source));
      // Display the visualization before like/copy controls in real chat DOM.
      const actions=typeof message.querySelector==="function"
        ? message.querySelector(".message-actions") : null;
      if(actions && typeof message.insertBefore==="function")
        message.insertBefore(widget,actions);
      else message.append(widget);
      return true;
    }catch{return false;} // Degrade to the unmodified answer text.
  }
  if(typeof window!=="undefined")window.NeralDynamicUI={plan,mount,calculate};
})();
