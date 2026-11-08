"use strict";
/*
 * NERAL Dynamic UI v1: declarative content only. No eval, Function,
 * arbitrary HTML, remote widgets, CSS from the model or model-generated JS.
 * Pure functions work in Node and the local browser.
 */
(function(root){
  const TYPES=new Set(["calculator","chart","table","quiz","simulation"]);
  const FUNCS={abs:1,sqrt:1,sin:1,cos:1,min:2,max:2,pow:2};
  const isObj=v=>v!==null&&typeof v==="object"&&!Array.isArray(v);
  const finite=v=>typeof v==="number"&&Number.isFinite(v);
  const word=(v,max=90)=>typeof v==="string"&&v.trim().length>0&&v.length<=max&&
    !/[\u0000-\u001f\u007f]/.test(v);
  const safeNumber=v=>finite(v)&&Math.abs(v)<=1e16;

  function evaluate(expression,variables={}){
    if(typeof expression!=="string"||expression.length<1||expression.length>240||
       !isObj(variables))return null;
    try{
      const tokens=[],pattern=/\s*((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?|[a-zA-Z_][a-zA-Z0-9_]*|[()+\-*/^,])/y;
      let pos=0;
      while(pos<expression.length){
        pattern.lastIndex=pos;
        const match=pattern.exec(expression);
        if(!match)throw Error("Invalid expression");
        tokens.push(match[1]);pos=pattern.lastIndex;
        if(tokens.length>128)throw Error("Expression too long");
      }
      let index=0;
      function peek(){return tokens[index];}
      function take(value){if(peek()===value){index++;return true;}return false;}
      function need(value){if(!take(value))throw Error("Invalid expression syntax");}
      function bounded(value){if(!safeNumber(value))throw Error("Math domain");return value;}
      function primary(depth){
        if(depth>20)throw Error("Nesting limit");
        const token=peek();
        if(take("(")){const value=add(depth+1);need(")");return value;}
        if(typeof token!=="string")throw Error("Missing token");
        if(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?$/.test(token)){
          index++;return bounded(Number(token));
        }
        if(/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(token)){
          index++;
          if(take("(")){
            if(!Object.prototype.hasOwnProperty.call(FUNCS,token))throw Error("Unknown function");
            const args=[add(depth+1)];
            while(take(","))args.push(add(depth+1));
            need(")");
            if(args.length!==FUNCS[token])throw Error("Wrong arity");
            const [x,y]=args;
            if(token==="abs")return bounded(Math.abs(x));
            if(token==="sqrt")return bounded(Math.sqrt(x));
            if(token==="sin")return bounded(Math.sin(x));
            if(token==="cos")return bounded(Math.cos(x));
            if(token==="min")return bounded(Math.min(x,y));
            if(token==="max")return bounded(Math.max(x,y));
            if(token==="pow")return bounded(Math.pow(x,y));
          }
          if(!Object.prototype.hasOwnProperty.call(variables,token)||
             !safeNumber(variables[token]))throw Error("Unknown variable");
          return variables[token];
        }
        throw Error("Unsupported token");
      }
      function power(depth){
        const left=primary(depth);
        if(take("^"))return bounded(Math.pow(left,unary(depth+1)));
        return left;
      }
      function unary(depth){
        if(depth>20)throw Error("Nesting limit");
        if(take("+"))return unary(depth+1);
        if(take("-"))return bounded(-unary(depth+1));
        return power(depth);
      }
      function multiply(depth){
        let left=unary(depth);
        while(peek()==="*"||peek()==="/"){
          const op=tokens[index++],right=unary(depth);
          if(op==="/"&&Math.abs(right)<1e-15)throw Error("Division by zero");
          left=bounded(op==="*"?left*right:left/right);
        }
        return left;
      }
      function add(depth){
        let left=multiply(depth);
        while(peek()==="+"||peek()==="-"){
          const op=tokens[index++],right=multiply(depth);
          left=bounded(op==="+"?left+right:left-right);
        }
        return left;
      }
      const result=add(0);
      if(index!==tokens.length)throw Error("Trailing token");
      return bounded(result);
    }catch{return null;}
  }

  function validate(spec){
    if(!isObj(spec)||!TYPES.has(spec.type)||!word(spec.title,100))return null;
    const base={type:spec.type,title:spec.title.trim()};
    if(spec.type==="calculator"){
      if(!Array.isArray(spec.inputs)||spec.inputs.length<1||spec.inputs.length>5||
         !word(spec.expression,240)||!word(spec.resultLabel,70))return null;
      const seen=new Set(),inputs=[];
      for(const v of spec.inputs){
        if(!isObj(v)||typeof v.id!=="string"||
           !/^[a-z][a-z0-9_]{0,12}$/.test(v.id)||
           seen.has(v.id)||!word(v.label,60)||
           (v.unit!==undefined&&v.unit!==""&&!word(v.unit,16))||
           ![v.min,v.max,v.step,v.value].every(finite)||
           Math.abs(v.min)>1e7||Math.abs(v.max)>1e7||
           !(v.min<v.max)||v.step<=0||v.step>v.max-v.min||
           v.value<v.min||v.value>v.max) return null;
        seen.add(v.id);
        inputs.push({id:v.id,label:v.label.trim(),unit:v.unit||"",
          min:v.min,max:v.max,step:v.step,value:v.value});
      }
      const precision=spec.precision===undefined?3:spec.precision;
      if(!Number.isInteger(precision)||precision<0||precision>6||
         (spec.unit!==undefined&&spec.unit!==""&&!word(spec.unit,16)))return null;
      const expression=spec.expression.trim(),vars=Object.fromEntries(inputs.map(x=>[x.id,x.value]));
      if(evaluate(expression,vars)===null)return null;
      return {...base,inputs,expression,resultLabel:spec.resultLabel.trim(),
              unit:spec.unit||"",precision};
    }
    if(spec.type==="chart"){
      if(!Array.isArray(spec.labels)||!Array.isArray(spec.values)||
         spec.labels.length<2||spec.labels.length>12||
         spec.labels.length!==spec.values.length||
         spec.labels.some(v=>!word(v,40))||
         spec.values.some(v=>!finite(v)||v<0||v>1e9)||
         (spec.unit!==undefined&&spec.unit!==""&&!word(spec.unit,24))||
         (spec.mode!==undefined&&!["bar","line"].includes(spec.mode)))return null;
      return {...base,labels:spec.labels.map(x=>x.trim()),values:spec.values.slice(),
        unit:spec.unit||"",mode:spec.mode||"bar"};
    }
    if(spec.type==="table"){
      if(!Array.isArray(spec.headers)||spec.headers.length<2||spec.headers.length>8||
         spec.headers.some(x=>!word(x,55))||!Array.isArray(spec.rows)||
         spec.rows.length<1||spec.rows.length>40||
         spec.rows.some(row=>!Array.isArray(row)||row.length!==spec.headers.length||
           row.some(x=>typeof x!=="string"||x.length>160||
             /[\u0000-\u001f\u007f]/.test(x))))return null;
      return {...base,headers:spec.headers.map(x=>x.trim()),rows:spec.rows.map(r=>r.slice())};
    }
    if(spec.type==="quiz"){
      if(!Array.isArray(spec.questions)||spec.questions.length<1||
         spec.questions.length>6)return null;
      const questions=[];
      for(const q of spec.questions){
        if(!isObj(q)||!word(q.prompt,240)||!Array.isArray(q.options)||
           q.options.length<2||q.options.length>5||
           q.options.some(x=>!word(x,140))||
           new Set(q.options).size!==q.options.length||
           !Number.isInteger(q.answer)||q.answer<0||q.answer>=q.options.length||
           (q.explanation!==undefined&&q.explanation!==""&&!word(q.explanation,300)))
          return null;
        questions.push({prompt:q.prompt.trim(),options:q.options.map(x=>x.trim()),
          answer:q.answer,explanation:q.explanation||""});
      }
      return {...base,questions};
    }
    if(spec.type==="simulation"){
      if(spec.preset!=="projectile")return null;
      return {...base,preset:"projectile"};
    }
    return null;
  }

  function extract(source){
    const text=typeof source==="string"?source:"";
    if(text.length>32768)return {text,widget:null};
    const pattern=/\x60{3}neral-ui\s*\r?\n([\s\S]{1,8192}?)\r?\n\x60{3}/i;
    const match=pattern.exec(text);
    if(!match)return {text,widget:null};
    let candidate;
    try{candidate=JSON.parse(match[1]);}catch{return {text,widget:null};}
    const widget=validate(candidate);
    if(!widget)return {text,widget:null};
    const prose=(text.slice(0,match.index)+text.slice(match.index+match[0].length))
      .replace(/\n{3,}/g,"\n\n").trim();
    return {text:prose,widget};
  }

  function intentFor(text){
    if(typeof text!=="string"||text.length>4000)return null;
    const v=text.trim().toLowerCase();
    if(!v)return null;
    if(/\b(quiz|abfragen|multiple.choice|lernfragen|wissensfragen)\b/i.test(v))return "quiz";
    if(/\b(simulier\w*|simulation|wurfparabel|schräger wurf|projektil)\b/i.test(v))return "simulation";
    if(/\b(diagramm|balken(?:diagramm)?|kurvendiagramm|grafik|graph|plot|chart|visualisier\w*)\b/i.test(v))return "chart";
    if(/\b(rechner|berechne interaktiv|kalkulator|schieberegler|slider|coulomb)\b/i.test(v))return "calculator";
    if(/\b(tabelle|tabellarisch|vergleichstabelle)\b/i.test(v))return "table";
    return null;
  }

  function fallbackFor(question){
    if(typeof question!=="string")return null;
    const v=question.toLowerCase(),intent=intentFor(question);
    if((intent==="calculator"||intent==="chart")&&/\bcoulomb\b/i.test(v)){
      return validate({type:"calculator",title:"Coulomb-Kraft",
        inputs:[
          {id:"q1",label:"Ladung 1",unit:"µC",min:-10,max:10,step:0.5,value:1},
          {id:"q2",label:"Ladung 2",unit:"µC",min:-10,max:10,step:0.5,value:1},
          {id:"r",label:"Abstand",unit:"m",min:0.1,max:3,step:0.05,value:0.3}
        ],expression:"0.0089875517923*abs(q1*q2)/(r^2)",
        resultLabel:"Betrag der Kraft",unit:"N",precision:4});
    }
    if(intent==="calculator"&&/\b(prozent|rabatt|mehrwertsteuer)\b/i.test(v)){
      return validate({type:"calculator",title:"Prozentrechner",
        inputs:[
          {id:"grundwert",label:"Grundwert",unit:"",min:0,max:10000,step:10,value:200},
          {id:"satz",label:"Prozentsatz",unit:"%",min:0,max:100,step:1,value:15}
        ],expression:"grundwert*satz/100",resultLabel:"Prozentwert",unit:"",precision:2});
    }
    if(intent==="simulation"&&/\b(wurf|projektil|wurfparabel)\b/i.test(v))
      return validate({type:"simulation",preset:"projectile",title:"Schräger Wurf"});
    return null;
  }

  const api={evaluate,validate,extract,intentFor,fallbackFor};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(root)root.NERALDynamicUI=api;
})(typeof globalThis!=="undefined"?globalThis:this);
