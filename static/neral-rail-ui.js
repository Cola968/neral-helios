"use strict";
/* BahnConnections inside NERAL answers. No untrusted HTML, no speculative paths. */
(function(root){
  const isObj=v=>v!==null&&typeof v==="object"&&!Array.isArray(v);
  const validTime=x=>typeof x==="string"&&x.length<60&&Number.isFinite(Date.parse(x));
  const point=p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&Math.abs(p[0])<=90&&Math.abs(p[1])<=180;
  const stop=s=>isObj(s)&&typeof s.name==="string"&&s.name.length>0&&s.name.length<=100;
  function validate(raw){
    if(!isObj(raw)||!["ok","empty"].includes(raw.status)||typeof raw.from!=="string"||
      !raw.from||raw.from.length>110||typeof raw.to!=="string"||!raw.to||raw.to.length>110||
      !Array.isArray(raw.journeys)||raw.journeys.length>5||
      typeof raw.source!=="string"||raw.source.length>140)return null;
    for(const j of raw.journeys){
      if(!isObj(j)||!validTime(j.startTime)||!validTime(j.endTime)||
        Date.parse(j.startTime)>=Date.parse(j.endTime)||
        !Number.isInteger(j.transfers)||j.transfers<0||j.transfers>8||
        !Array.isArray(j.legs)||!j.legs.length||j.legs.length>10)return null;
      for(const l of j.legs){
        if(!isObj(l)||typeof l.name!=="string"||l.name.length>100||
          !stop(l.from)||!stop(l.to)||!validTime(l.startTime)||!validTime(l.endTime)||
          !Array.isArray(l.points)||l.points.length>2000||!l.points.every(point))return null;
      }
    }
    return raw;
  }
  function parse(input){
    if(typeof input!=="string"||input.length>500)return null;
    const text=input.replace(/\s+/g," ").trim();
    if(!/\b(?:zug(?:verbindung)?|bahn(?:verbindung|fahrt)?|verbindung(?:en)?|reiseverbindung|fahrplan)\b/i.test(text))return null;
    const match=/\bvon\s+(.+?)\s+nach\s+(.+)$/i.exec(text);
    if(!match)return null;
    const from=match[1].trim().replace(/[.!?]+$/,"");
    let to=match[2].trim();
    if(/\bam\s+\d{1,2}[.\/]/i.test(to))return null;
    const day=/\b(übermorgen|morgen|heute)\b/i.exec(to);
    const time=/\b(?:um|ab)\s*(\d{1,2})(?::(\d{2}))?\s*(?:uhr)?\b/i.exec(to);
    to=to.split(/\s+\b(?:übermorgen|morgen|heute|um|ab|am)\b/i)[0]
      .trim().replace(/[.!?.,]+$/,"").trim();
    if(!from||!to||from.length>80||to.length>80||/[<>{}\\\/]|https?:|www\./i.test(from+" "+to))
      return null;
    const result={from,to};
    if(day)result.when=day[1].toLowerCase();
    if(time){
      const h=Number(time[1]),m=Number(time[2]||"0");
      if(h>23||m>59)return null;
      result.time=String(h).padStart(2,"0")+":"+String(m).padStart(2,"0");
    }
    return result;
  }
  function distance(a,b){
    const r=Math.PI/180,dy=(b[0]-a[0])*r,dx=(b[1]-a[1])*r;
    const h=Math.sin(dy/2)**2+Math.cos(a[0]*r)*Math.cos(b[0]*r)*Math.sin(dx/2)**2;
    return 12742*Math.asin(Math.min(1,Math.sqrt(h)));
  }
  function geometrySegments(points){
    if(!Array.isArray(points))return [];
    const result=[];let current=[];
    const flush=()=>{if(current.length>1)result.push(current);current=[];};
    for(const p of points){
      if(!point(p)){flush();continue;}
      if(current.length&&distance(current[current.length-1],p)>5)flush();
      current.push(p);
    }
    flush();return result;
  }
  const api={parse,validate,geometrySegments};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(typeof document==="undefined"||!root)return;
  const ns="http://www.w3.org/2000/svg";
  function el(tag,cls,value){
    const node=document.createElement(tag);if(cls)node.className=cls;
    if(value!==undefined)node.textContent=String(value);return node;
  }
  function svg(tag,attrs={}){
    const node=document.createElementNS(ns,tag);
    for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));
    return node;
  }
  function clock(t){
    return validTime(t)?new Intl.DateTimeFormat("de-DE",{timeZone:"Europe/Berlin",
      hour:"2-digit",minute:"2-digit"}).format(new Date(t)):"–";
  }
  function duration(seconds){
    const m=Math.round(Math.max(0,Number(seconds)||0)/60);
    return Math.floor(m/60)+" Std. "+String(m%60).padStart(2,"0")+" Min.";
  }
  function state(j){
    return j.cancelled?"Ausfall":j.realtimeStatus==="live"?"Echtzeit":
      j.realtimeStatus==="schedule"?"Nur Fahrplan":"Echtzeit teilweise";
  }
  function drawMap(into,journey){
    const box=el("div","rail-map"),top=el("div","rail-map-head");
    top.append(el("strong",null,"Streckenkarte"),el("span","rail-small","Aus Streckendaten"));
    box.append(top);
    const segments=journey.legs.flatMap(l=>geometrySegments(l.points));
    const stops=[journey.legs[0].from,journey.legs[journey.legs.length-1].to];
    const pins=stops.filter(s=>Number.isFinite(s.lat)&&Number.isFinite(s.lon));
    const all=segments.flat().concat(pins.map(s=>[s.lat,s.lon])).filter(point);
    if(!all.length){
      box.append(el("p","rail-map-missing","Keine geografischen Streckendaten verfügbar."));
      into.append(box);return;
    }
    const minLat=Math.min(...all.map(p=>p[0])),maxLat=Math.max(...all.map(p=>p[0]));
    const minLon=Math.min(...all.map(p=>p[1])),maxLon=Math.max(...all.map(p=>p[1]));
    const scaleLon=Math.max(.05,Math.cos((minLat+maxLat)/2*Math.PI/180));
    const extentX=Math.max(.04,(maxLon-minLon)*scaleLon),extentY=Math.max(.04,maxLat-minLat);
    const scale=Math.min(580/extentX,235/extentY);
    const x0=(640-extentX*scale)/2,y0=(290-extentY*scale)/2;
    const xy=p=>[x0+(p[1]-minLon)*scaleLon*scale,y0+(maxLat-p[0])*scale];
    const canvas=svg("svg",{viewBox:"0 0 640 290",role:"img",
      "aria-label":"Karte nur mit vorhandenen Bahn-Streckenpunkten"});
    for(const path of segments){
      const d=path.map((p,i)=>{
        const [x,y]=xy(p);return (i?"L":"M")+x.toFixed(1)+" "+y.toFixed(1);
      }).join(" ");
      canvas.append(svg("path",{d,stroke:"currentColor",fill:"none",
        "stroke-width":2.4,"stroke-linejoin":"round","stroke-linecap":"round"}));
    }
    for(const [i,s] of stops.entries()){
      if(!Number.isFinite(s.lat)||!Number.isFinite(s.lon))continue;
      const [x,y]=xy([s.lat,s.lon]);
      canvas.append(svg("circle",{cx:x,cy:y,r:5,stroke:"var(--bg,#fff)",
        "stroke-width":2,fill:"var(--rail-pin,#303030)"}));
      const label=svg("text",{x:x+(i===0?10:-10),y:y+(i===0?-9:18),
        "text-anchor":i===0?"start":"end","font-size":13,fill:"currentColor"});
      label.textContent=s.name.slice(0,22);canvas.append(label);
    }
    box.append(canvas);
    box.append(el("p","rail-small",segments.length?
      (segments.length>1?"Fehlende Streckenabschnitte werden nicht verbunden.":"Original-Streckenpunkte des Datenanbieters.") :
      "Bahnhofspositionen vorhanden, aber kein Streckenverlauf. Keine Luftlinie gezeichnet."));
    into.append(box);
  }
  function legsView(j){
    const wrap=el("section","rail-legs");
    wrap.append(el("h4",null,"Fahrtverlauf"));
    j.legs.forEach((leg,i)=>{
      const row=el("div","rail-leg");
      row.append(el("strong",null,leg.name));
      if(leg.cancelled)row.append(el("span","rail-cancel","Ausfall"));
      const from=el("div","rail-stop");
      from.append(el("b",null,clock(leg.startTime)),el("span",null,leg.from.name));
      if(leg.from.track)from.append(el("small",null,"Gleis "+leg.from.track));
      const to=el("div","rail-stop");
      to.append(el("b",null,clock(leg.endTime)),el("span",null,leg.to.name));
      if(leg.to.track)to.append(el("small",null,"Gleis "+leg.to.track));
      row.append(from,to);
      if(leg.realtime&&validTime(leg.scheduledStartTime)){
        const d=Math.round((Date.parse(leg.startTime)-Date.parse(leg.scheduledStartTime))/60000);
        if(d)row.append(el("small",d>=5?"rail-cancel":"rail-small",
          (d>0?"+":"")+d+" Min. Abfahrtsprognose"));
      }
      wrap.append(row);
      const next=j.legs[i+1];
      if(next){
        const wait=Math.round((Date.parse(next.startTime)-Date.parse(leg.endTime))/60000);
        wrap.append(el("div","rail-transfer",leg.to.name+" · "+Math.max(0,wait)+" Min. Umstieg"));
      }
    });
    if(Array.isArray(j.warnings))for(const warning of j.warnings.slice(0,2))
      wrap.append(el("p","rail-warning",warning));
    return wrap;
  }
  function render(container,raw){
    const data=validate(raw);if(!data)return null;
    const card=el("section","neral-rail-ui");
    card.setAttribute("aria-label","BahnConnections · Reiseverbindungen");
    const head=el("header","rail-head");
    const identity=el("div","rail-identity");
    identity.append(el("strong",null,"BahnConnections"),el("span","rail-small",
      data.realtimeStatus==="live"?"Echtzeitdaten":"Fahrplanauskunft"));
    head.append(identity);
    head.append(el("h3",null,data.from+" → "+data.to));
    head.append(el("p","rail-small",data.source+
      (validTime(data.updatedAt)?" · Stand "+clock(data.updatedAt):"")));
    card.append(head);
    if(!data.journeys.length)card.append(el("p","rail-map-missing",
      "Keine bestätigte Verbindung für diese Suche gefunden."));
    else{
      const layout=el("div","rail-layout"),list=el("div","rail-results"),detail=el("div","rail-detail");
      const buttons=[];
      function choose(index){
        buttons.forEach((b,i)=>b.setAttribute("aria-pressed",String(i===index)));
        detail.replaceChildren();
        const j=data.journeys[index];
        const summary=el("div","rail-journey-meta");
        summary.append(el("span",null,duration(j.durationSeconds)));
        summary.append(el("span",null,j.transfers?
          j.transfers+" Umstieg"+(j.transfers===1?"":"e"):"Direkt"));
        summary.append(el("span",j.cancelled?"rail-cancel":"",state(j)));
        detail.append(summary);
        drawMap(detail,j);detail.append(legsView(j));
      }
      data.journeys.forEach((j,i)=>{
        const button=el("button","rail-result");button.type="button";
        const times=el("div","rail-result-times");
        times.append(el("strong",null,clock(j.startTime)),
          el("span",null,"–"),el("strong",null,clock(j.endTime)));
        button.append(times,el("span","rail-small",duration(j.durationSeconds)+
          " · "+(j.transfers?j.transfers+" Umstieg"+(j.transfers===1?"":"e"):"Direkt")));
        button.append(el("small",j.cancelled?"rail-cancel":"rail-small",state(j)));
        button.addEventListener("click",()=>choose(i));
        buttons.push(button);list.append(button);
      });
      layout.append(list,detail);card.append(layout);choose(0);
    }
    if(Array.isArray(data.warnings)&&data.warnings.length)
      card.append(el("p","rail-warning",data.warnings.slice(0,2).join(" · ")));
    const footer=el("footer","rail-footer");
    footer.append(el("span",null,"Fahrplandaten können sich ändern. Vor Abfahrt prüfen."));
    const link=el("a",null,"In BahnConnections öffnen");
    link.href="https://bahnconnections-de.a-stad.chatgpt.site/";
    link.target="_blank";link.rel="noopener noreferrer";
    footer.append(link);card.append(footer);
    container.append(card);return card;
  }
  function summary(raw){
    const d=validate(raw);
    if(!d)return "";
    const first=d.journeys[0];
    return first?(d.from+" nach "+d.to+": "+clock(first.startTime)+" bis "+
      clock(first.endTime)+", "+first.transfers+" Umstiege, "+duration(first.durationSeconds)+
      "; BahnConnections, Angaben ohne Gewähr."):
      ("Keine bestätigte Verbindung von "+d.from+" nach "+d.to+" gefunden.");
  }
  api.render=render;api.summary=summary;
  root.NERALJourneyUI=api;
})(typeof globalThis!=="undefined"?globalThis:this);
