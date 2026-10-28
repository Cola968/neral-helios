'use strict';
// No HTML parsing: model output and saved messages are always untrusted text.
(function(root){
  const bytes=s=>new TextEncoder().encode(s).length;
  function context(messages,prompt,budget,maxContext,checkpoint){
    const pairs=[];
    for(let i=0;i+1<messages.length;i++){
      const u=messages[i],a=messages[i+1];
      if(u.role==='user'&&u.mode==='chat'&&a.role==='assistant'&&a.mode==='trained-generation'&&a.checkpoint===checkpoint&&
         !a.interrupted&&!a.imported&&!u.imported&&(!a.source||a.source==='native-model')){
        pairs.push([{role:'user',content:u.content},{role:'assistant',content:a.content}]);i++;
      }
    }
    const format=rows=>rows.map(m=>(m.role==='user'?'User':'Assistant')+': '+m.content.trim()+'\n').join('')+'User: '+prompt.trim()+'\nAssistant:';
    const original=pairs.length;
    while(pairs.length&&(pairs.length>6||bytes(format(pairs.flat()))+1+budget>maxContext))pairs.shift();
    const history=pairs.flat();
    return {history,omitted:original-pairs.length,fits:bytes(format(history))+1+budget<=maxContext};
  }
  function sourceLabel(message,labels={}){
    if(message.mode==='external-model'||String(message.source||'').startsWith('external'))return 'Externes Modell · nicht NERAL';
    if(message.mode==='solve-tool')return 'Rechenwerkzeug · '+(message.source==='tool:calculator'?'Rechner':message.source==='tool:percentage'?'Prozentrechnung':'SOLVE');
    if(message.mode==='code-tool')return 'Codeprüfung · lokal, ohne Ausführung';
    if(message.mode==='guarded-response')return 'Sicherheitsregel · keine Modellantwort';
    if(message.mode==='error')return 'Anfrage fehlgeschlagen';
    if(message.mode==='no-model')return 'Kein natives Modell geladen';
    const names={original:'Neral 20M · Original',pilot_01:'Neral 20M · Pilot 01',pilot_02:'Neral 20M · Preview 02',neral40m:'Neral 40M · Experiment',neral50m:'Neral 50M · Sprachpilot',neral50m_i1:'Neral 50M · I1'};
    const name=names[message.checkpoint]||labels[message.checkpoint];
    return message.mode==='trained-generation'&&name?'Eigenes Modell · '+name:'Modellherkunft nicht verifiziert';
  }
  function generationResult(result){
    if(!result||typeof result.text!=='string')throw Error('Der Server hat keine gültige Antwort gesendet.');
    if(result.mode==='trained-generation'){
      if(result.model_generated!==true||typeof result.active_checkpoint!=='string'||!result.active_checkpoint||
         (result.source&&result.source!=='native-model'))throw Error('Die Herkunft der Modellantwort ist widersprüchlich.');
    }else if(!['solve-tool','code-tool','guarded-response'].includes(result.mode)||result.model_generated!==false){
      throw Error('Die Herkunft der Antwort ist nicht verifiziert.');
    }
    const sources={'solve-tool':['tool:calculator','tool:percentage','tool:solve'],'code-tool':['tool:code-check'],'guarded-response':['local-rule:security']};
    if(sources[result.mode]&&result.source&&!sources[result.mode].includes(result.source))throw Error('Die Herkunft der Werkzeugantwort ist widersprüchlich.');
    return result;
  }
  async function readGenerationResponse(response,onProgress){
    if(!response.headers.get('content-type')?.includes('application/x-ndjson')){
      let value;try{value=await response.json();}catch{throw Error('Das Antwortformat ist nicht lesbar. Bitte erneut versuchen.');}
      return generationResult(value);
    }
    if(!response.body)throw Error('Der Server hat keinen Antwortstrom gesendet.');
    const reader=response.body.getReader(),decoder=new TextDecoder('utf-8',{fatal:true});
    let buffer='',result=null,total=0;
    const consume=line=>{
      if(!line.trim())return;
      let item;try{item=JSON.parse(line);}catch{throw Error('Das Antwortformat ist nicht lesbar. Bitte erneut versuchen.');}
      if(!item||result!==null)throw Error('Ungültiger Abschluss des Antwortstroms.');
      if(item.type==='done')result=generationResult(item.result);
      else if(item.type==='error')throw Error(typeof item.error==='string'?item.error:'Die Berechnung wurde unterbrochen.');
      else if(item.type==='progress'){
        if(typeof item.text!=='string'||!Number.isInteger(item.generated_tokens)||item.generated_tokens<0)throw Error('Ungültige Teilantwort.');
        onProgress(item);
      }else if(!['start','heartbeat'].includes(item.type))throw Error('Ungültiges Antwortformat.');
    };
    try{
      while(true){
        const {value,done}=await reader.read();total+=value?.length||0;
        if(total>2*1024*1024)throw Error('Antwortstrom überschreitet das Größenlimit.');
        buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
        if(buffer.length>262144)throw Error('Ungültiger Antwortstrom.');
        let end;while((end=buffer.indexOf('\n'))>=0){consume(buffer.slice(0,end));buffer=buffer.slice(end+1);}
        if(done){consume(buffer);break;}
      }
      if(result===null)throw Error('Verbindung unterbrochen. Die Teilantwort ist unvollständig.');
      return result;
    }finally{await reader.cancel().catch(()=>{});reader.releaseLock();}
  }
  function blocks(text){
    const out=[];let code=null,language='';
    for(const line of String(text).split('\n')){
      if(line.startsWith('```')){
        if(code!==null){out.push({kind:'code',text:code.join('\n'),language});code=null;}
        else{code=[];language=line.slice(3).trim().replace(/[^a-zA-Z0-9_+.-]/g,'').slice(0,24);}
      }else if(code!==null)code.push(line);
      else out.push({kind:'text',text:line});
    }
    if(code!==null)out.push({kind:'code',text:code.join('\n'),language});
    return out;
  }
  function inline(parent,text,doc){
    for(const part of text.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g)){
      const tag=part.startsWith('`')&&part.endsWith('`')?'code':part.startsWith('**')&&part.endsWith('**')?'strong':null;
      if(!tag){parent.append(doc.createTextNode(part));continue;}
      const n=doc.createElement(tag);n.textContent=part.slice(tag==='code'?1:2,tag==='code'?-1:-2);parent.append(n);
    }
  }
  function render(parent,text,doc=document){
    parent.replaceChildren();
    for(const block of blocks(text)){
      if(block.kind==='code'){
        const wrap=doc.createElement('div');wrap.className='code-block';
        const bar=doc.createElement('div');bar.className='code-toolbar';
        const label=doc.createElement('span');label.textContent=block.language||'Code';
        const copy=doc.createElement('button');copy.type='button';copy.textContent='Code kopieren';
        copy.onclick=async()=>{try{await navigator.clipboard.writeText(block.text);copy.textContent='Kopiert';}catch{copy.textContent='Bitte Text markieren';}};
        bar.append(label,copy);const pre=doc.createElement('pre'),code=doc.createElement('code');code.textContent=block.text;pre.append(code);wrap.append(bar,pre);parent.append(wrap);
      }else{
        const heading=block.text.match(/^(#{1,3})\s+(.+)$/),bullet=block.text.match(/^[-*]\s+(.+)$/);
        const n=doc.createElement(heading?'h'+(heading[1].length+2):'div');n.className='markdown-line';
        inline(n,heading?heading[2]:bullet?'• '+bullet[1]:block.text||'\u00a0',doc);parent.append(n);
      }
    }
  }
  function parseImport(text){
    if(bytes(text)>2*1024*1024)throw Error('Import ist größer als 2 MB.');
    const data=JSON.parse(text);
    if(data?.format!=='neral-chat-v1'||typeof data.title!=='string'||!Array.isArray(data.messages)||data.messages.length>500)throw Error('Ungültiges Chat-Exportformat.');
    const modes=new Set(['chat','solve','code','trained-generation','solve-tool','code-tool','guarded-response','external-model','error','no-model']);
    const messages=data.messages.map(m=>{
      if(!m||!['user','assistant'].includes(m.role)||typeof m.content!=='string'||bytes(m.content)>32768||!modes.has(m.mode))throw Error('Ungültige Nachricht im Import.');
      return {role:m.role,content:m.content,mode:m.mode,checkpoint:typeof m.checkpoint==='string'?m.checkpoint.slice(0,64):undefined,
        source:typeof m.source==='string'?m.source.slice(0,80):undefined,imported:true,interrupted:m.interrupted===true,meta:'Importierter Verlauf · nicht neu berechnet'};
    });
    return {title:data.title.slice(0,80)||'Importierter Chat',messages,draft:'',temporary:false,imported:true};
  }
  const api={context,blocks,render,parseImport,sourceLabel,readGenerationResponse};
  if(typeof module!=='undefined')module.exports=api;else root.HeliosCore=api;
})(typeof window==='undefined'?{}:window);
