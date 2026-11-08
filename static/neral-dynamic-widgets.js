"use strict";
/* Safe DOM renderer for NERAL Dynamic UI v1. No model-provided markup or scripts. */
(function(root){
  const api=root&&root.NERALDynamicUI;
  if(!api||typeof document==="undefined")return;
  const SVG_NS="http://www.w3.org/2000/svg";
  const make=(tag,cls,text)=>{
    const node=document.createElement(tag);
    if(cls)node.className=cls;
    if(text!==undefined)node.textContent=String(text);
    return node;
  };
  const svg=(tag,attrs={})=>{
    const node=document.createElementNS(SVG_NS,tag);
    for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));
    return node;
  };
  function labelFor(label,input){
    const wrapper=make("label","neral-ui-label");
    wrapper.append(make("span",null,label),input);
    return wrapper;
  }
  const fmt=(value,digits=2)=>Number.isFinite(value)?
    value.toLocaleString("de-DE",{maximumFractionDigits:digits}):"–";
  const clamp=(v,min,max)=>Math.min(max,Math.max(min,v));
  function meter(name,value,unit){
    const box=make("div","neral-ui-meter");
    box.append(make("span","neral-ui-muted",name));
    const output=make("strong",null,fmt(value,3)+(unit?" "+unit:""));
    box.append(output);
    return {box,output};
  }
  function numberControl(config,onChange){
    const wrap=make("div","neral-ui-control");
    const caption=make("div","neral-ui-control-name",config.label+(config.unit?" · "+config.unit:""));
    const row=make("div","neral-ui-control-row");
    const range=make("input");
    range.type="range";range.min=String(config.min);range.max=String(config.max);
    range.step=String(config.step);range.value=String(config.value);
    range.setAttribute("aria-label",config.label+" Schieberegler");
    const number=make("input");
    number.type="number";number.min=String(config.min);number.max=String(config.max);
    number.step=String(config.step);number.value=String(config.value);
    number.setAttribute("aria-label",config.label+" Zahlenwert");
    row.append(range,number);wrap.append(caption,row);
    function apply(value,source){
      if(!Number.isFinite(value))return;
      const next=clamp(value,config.min,config.max);
      range.value=String(next);
      number.value=String(next);
      onChange(next);
    }
    range.addEventListener("input",()=>apply(Number(range.value),range));
    number.addEventListener("change",()=>apply(Number(number.value),number));
    return {wrap,range,number};
  }
  function calculator(body,spec){
    const values=Object.fromEntries(spec.inputs.map(x=>[x.id,x.value]));
    const grid=make("div","neral-ui-controls");
    const result=meter(spec.resultLabel,api.evaluate(spec.expression,values),spec.unit);
    result.box.setAttribute("role","status");
    result.box.setAttribute("aria-live","polite");
    function update(){
      const value=api.evaluate(spec.expression,values);
      result.output.textContent=value===null
        ?"Für diese Werte nicht definiert"
        :fmt(value,spec.precision)+(spec.unit?" "+spec.unit:"");
    }
    for(const config of spec.inputs){
      const control=numberControl(config,value=>{values[config.id]=value;update();});
      grid.append(control.wrap);
    }
    const formula=make("p","neral-ui-formula","Formel: "+spec.expression);
    body.append(grid,result.box,formula);
    update();
  }

  function chart(body,spec){
    const actions=make("div","neral-ui-chart-actions");
    const stage=make("div","neral-ui-chart-stage");
    let mode=spec.mode;
    const b1=make("button","", "Balken"),b2=make("button","", "Linie");
    b1.type="button";b2.type="button";
    actions.append(b1,b2);
    function render(){
      stage.replaceChildren();
      b1.setAttribute("aria-pressed",String(mode==="bar"));
      b2.setAttribute("aria-pressed",String(mode==="line"));
      if(mode==="bar"){
        const max=Math.max(1,...spec.values);
        for(let i=0;i<spec.values.length;i++){
          const row=make("div","neral-ui-bar-row");
          const name=make("span","neral-ui-bar-name",spec.labels[i]);
          const track=make("div","neral-ui-bar-track");
          const bar=make("div","neral-ui-bar-fill");
          bar.style.width=(100*spec.values[i]/max).toFixed(2)+"%";
          track.append(bar);
          const val=make("span","neral-ui-bar-value",fmt(spec.values[i],2)+(spec.unit?" "+spec.unit:""));
          row.append(name,track,val);stage.append(row);
        }
      }else{
        const canvas=svg("svg",{viewBox:"0 0 560 250",role:"img",
          "aria-label":"Liniendiagramm: "+spec.title});
        const max=Math.max(1,...spec.values);
        const pts=spec.values.map((v,i)=>{
          const x=40+480*i/(spec.values.length-1);
          const y=204-170*v/max;
          return [x,y];
        });
        canvas.append(svg("path",{d:"M40 20 V204 H530",fill:"none",
          stroke:"currentColor","stroke-width":1,opacity:0.35}));
        const path=svg("path",{d:pts.map(([x,y],i)=>
          (i?"L":"M")+x.toFixed(2)+" "+y.toFixed(2)).join(" "),
          fill:"none",stroke:"currentColor","stroke-width":2.5,
          "stroke-linecap":"round","stroke-linejoin":"round"});
        canvas.append(path);
        for(let i=0;i<pts.length;i++){
          const [x,y]=pts[i];
          const dot=svg("circle",{cx:x,cy:y,r:4,fill:"currentColor"});
          const title=svg("title");title.textContent=spec.labels[i]+": "+fmt(spec.values[i],2)+
            (spec.unit?" "+spec.unit:"");
          dot.append(title);canvas.append(dot);
          const label=svg("text",{x, y:227, "text-anchor":"middle","font-size":10,
            fill:"currentColor"});
          label.textContent=spec.labels[i].slice(0,12);
          canvas.append(label);
        }
        stage.append(canvas);
        const values=make("p","neral-ui-muted",spec.labels.map((label,i)=>
          label+": "+fmt(spec.values[i],2)).join(" · "));
        stage.append(values);
      }
    }
    b1.addEventListener("click",()=>{mode="bar";render();});
    b2.addEventListener("click",()=>{mode="line";render();});
    body.append(actions,stage);render();
  }

  function table(body,spec){
    const search=make("input","neral-ui-search");
    search.type="search";search.placeholder="Tabelle filtern";
    search.setAttribute("aria-label","Zeilen filtern");
    const status=make("p","neral-ui-muted");
    status.setAttribute("role","status");
    const wrap=make("div","neral-ui-table-wrap");
    const table=make("table","neral-ui-table");
    const head=make("thead"),tr=make("tr"),tbody=make("tbody");
    const headers=spec.headers.slice();
    let sort=-1,ascending=true;
    function refresh(){
      const q=search.value.trim().toLocaleLowerCase("de-DE");
      let rows=spec.rows.filter(row=>row.some(cell=>cell.toLocaleLowerCase("de-DE").includes(q)));
      if(sort>=0){
        rows=rows.slice().sort((a,b)=>{
          const x=a[sort],y=b[sort],nx=Number(x.replace(",",".")),ny=Number(y.replace(",","."));
          const delta=x.trim()&&y.trim()&&Number.isFinite(nx)&&Number.isFinite(ny)
            ?nx-ny:x.localeCompare(y,"de",{sensitivity:"base",numeric:true});
          return ascending?delta:-delta;
        });
      }
      tbody.replaceChildren();
      for(const row of rows){
        const line=make("tr");
        for(const cell of row){const td=make("td");td.textContent=cell;line.append(td);}
        tbody.append(line);
      }
      status.textContent=rows.length+" von "+spec.rows.length+" Zeilen";
    }
    headers.forEach((label,index)=>{
      const th=make("th"),button=make("button",null,label+" ↕");
      button.type="button";
      button.setAttribute("aria-label","Nach "+label+" sortieren");
      button.addEventListener("click",()=>{
        ascending=sort===index?!ascending:true;sort=index;refresh();
      });
      th.append(button);tr.append(th);
    });
    head.append(tr);table.append(head,tbody);wrap.append(table);
    search.addEventListener("input",refresh);
    body.append(search,status,wrap);
    refresh();
  }

  function quiz(body,spec){
    const progress=make("p","neral-ui-muted"),question=make("h4","neral-ui-question");
    const answers=make("div","neral-ui-answers");
    const feedback=make("p","neral-ui-feedback");
    feedback.setAttribute("role","status");
    feedback.setAttribute("aria-live","polite");
    const next=make("button","neral-ui-next","Nächste Frage");
    next.type="button";next.hidden=true;
    let index=0,score=0,answered=false;
    function render(){
      answered=false;answers.replaceChildren();feedback.textContent="";next.hidden=true;
      if(index>=spec.questions.length){
        progress.textContent="Quiz abgeschlossen";
        question.textContent=score+" von "+spec.questions.length+" richtig";
        next.textContent="Erneut versuchen";next.hidden=false;return;
      }
      const q=spec.questions[index];
      progress.textContent="Frage "+(index+1)+" / "+spec.questions.length+
        " · "+score+" Punkte";
      question.textContent=q.prompt;
      q.options.forEach((answer,i)=>{
        const b=make("button","neral-ui-answer",answer);
        b.type="button";
        b.addEventListener("click",()=>{
          if(answered)return;
          answered=true;
          if(i===q.answer)score++;
          feedback.textContent=(i===q.answer?"Richtig. ":"Nicht richtig. ")+
            (q.explanation||"");
          for(const button of answers.querySelectorAll("button"))button.disabled=true;
          b.setAttribute("data-selected","true");
          next.hidden=false;
          next.textContent=index===spec.questions.length-1?"Ergebnis":"Nächste Frage";
        });
        answers.append(b);
      });
    }
    next.addEventListener("click",()=>{
      if(index>=spec.questions.length){index=0;score=0;next.textContent="Nächste Frage";}
      else index++;
      render();
    });
    body.append(progress,question,answers,feedback,next);
    render();
  }

  function simulation(body){
    const params=[
      {id:"speed",label:"Startgeschwindigkeit",unit:"m/s",min:1,max:60,step:1,value:20},
      {id:"angle",label:"Wurfwinkel",unit:"°",min:5,max:85,step:1,value:45},
      {id:"gravity",label:"Schwerebeschleunigung",unit:"m/s²",min:1,max:20,step:0.01,value:9.81}
    ];
    const values=Object.fromEntries(params.map(x=>[x.id,x.value]));
    const grid=make("div","neral-ui-controls");
    const drawing=svg("svg",{viewBox:"0 0 620 248",role:"img",
      "aria-label":"Wurfparabel bei verschiedenen Geschwindigkeiten und Winkeln"});
    const axis=svg("path",{d:"M30 25 V215 H595",stroke:"currentColor","stroke-width":1,
      fill:"none",opacity:.38});
    const curve=svg("path",{fill:"none",stroke:"currentColor",
      "stroke-width":2.8,"stroke-linecap":"round"});
    const dot=svg("circle",{r:6,fill:"currentColor"});
    drawing.append(axis,curve,dot);
    const timeLabel=make("label","neral-ui-time-label","Zeitpunkt");
    const time=make("input");time.type="range";time.min="0";time.step="0.05";
    time.setAttribute("aria-label","Zeitpunkt während des Wurfs");
    const timeValue=make("span","neral-ui-muted");
    timeLabel.append(time,timeValue);
    const metrics=make("div","neral-ui-metrics");
    const range=meter("Reichweite",0,"m"),height=meter("Maximale Höhe",0,"m"),
          flight=meter("Flugzeit",0,"s");
    metrics.append(range.box,height.box,flight.box);
    const note=make("p","neral-ui-muted","Idealisierte Wurfparabel ohne Luftwiderstand.");
    function update(resetTime){
      const speed=values.speed,theta=values.angle*Math.PI/180,g=values.gravity;
      const vx=speed*Math.cos(theta),vy=speed*Math.sin(theta);
      const tEnd=2*vy/g,r=vx*tEnd,h=vy*vy/(2*g);
      range.output.textContent=fmt(r,2)+" m";
      height.output.textContent=fmt(h,2)+" m";
      flight.output.textContent=fmt(tEnd,2)+" s";
      time.max=String(tEnd);
      if(resetTime)time.value=String(tEnd/2);
      const t=clamp(Number(time.value)||0,0,tEnd);
      timeValue.textContent="t = "+fmt(t,2)+" s";
      const sx=x=>30+535*(x/Math.max(r,0.01));
      const sy=y=>215-175*(y/Math.max(h,0.01));
      const pieces=[];
      for(let i=0;i<=70;i++){
        const tt=tEnd*i/70;
        const x=vx*tt,y=Math.max(0,vy*tt-0.5*g*tt*tt);
        pieces.push((i?"L":"M")+sx(x).toFixed(2)+" "+sy(y).toFixed(2));
      }
      curve.setAttribute("d",pieces.join(" "));
      dot.setAttribute("cx",sx(vx*t).toFixed(2));
      dot.setAttribute("cy",sy(Math.max(0,vy*t-0.5*g*t*t)).toFixed(2));
    }
    for(const config of params){
      const control=numberControl(config,value=>{
        values[config.id]=value;update(true);
      });
      grid.append(control.wrap);
    }
    time.addEventListener("input",()=>update(false));
    body.append(grid,drawing,timeLabel,metrics,note);
    update(true);
  }

  function mount(message,spec){
    if(!message||typeof message.append!=="function")return null;
    const widget=api.validate(spec);
    if(!widget)return null;
    const card=make("section","neral-dynamic-widget");
    card.setAttribute("aria-label",widget.title);
    const header=make("div","neral-ui-head");
    header.append(make("strong",null,widget.title));
    header.append(make("span","neral-ui-muted","Interaktiv"));
    const body=make("div","neral-ui-body");
    if(widget.type==="calculator")calculator(body,widget);
    else if(widget.type==="chart")chart(body,widget);
    else if(widget.type==="table")table(body,widget);
    else if(widget.type==="quiz")quiz(body,widget);
    else if(widget.type==="simulation")simulation(body);
    const provenance=make("p","neral-ui-disclaimer",
      widget.type==="simulation"?"Lokale Physikberechnung, idealisiertes Modell.":
      "Angaben und Aufgaben aus der Modellantwort können fehlerhaft sein.");
    card.append(header,body,provenance);
    message.append(card);
    return card;
  }
  api.mount=mount;
})(typeof globalThis!=="undefined"?globalThis:this);
