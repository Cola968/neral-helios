'use strict';
// No HTML parsing: model output and saved messages are always untrusted text.
(function(root){
  const bytes=s=>new TextEncoder().encode(s).length;
  function context(messages,prompt,budget,maxContext,checkpoint){
    const pairs=[];
    for(let i=0;i+1<messages.length;i++){
      const u=messages[i],a=messages[i+1];
      if(u.role==='user'&&u.mode==='chat'&&a.role==='assistant'&&a.mode==='trained-generation'&&a.checkpoint===checkpoint){
        pairs.push([{role:'user',content:u.content},{role:'assistant',content:a.content}]);i++;
      }
    }
    const format=rows=>rows.map(m=>(m.role==='user'?'User':'Assistant')+': '+m.content.trim()+'\n').join('')+'User: '+prompt.trim()+'\nAssistant:';
    const original=pairs.length;
    while(pairs.length&&(pairs.length>6||bytes(format(pairs.flat()))+1+budget>maxContext))pairs.shift();
    const history=pairs.flat();
    return {history,omitted:original-pairs.length,fits:bytes(format(history))+1+budget<=maxContext};
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
    // Run only on inert text nodes already created by the safe Markdown parser.
    // KaTeX ignores <pre> and <code> so Python/LaTeX examples stay intact.
    if(typeof window!=="undefined")window.NeralMath?.render(parent);
  }
  function parseImport(text){
    if(bytes(text)>2*1024*1024)throw Error('Import ist größer als 2 MB.');
    const data=JSON.parse(text);
    if(data?.format!=='neral-chat-v1'||typeof data.title!=='string'||!Array.isArray(data.messages)||data.messages.length>500)throw Error('Ungültiges Chat-Exportformat.');
    const modes=new Set(['chat','solve','code','trained-generation','solve-tool','code-tool','guarded-response']);
    const messages=data.messages.map(m=>{
      if(!m||!['user','assistant'].includes(m.role)||typeof m.content!=='string'||bytes(m.content)>32768||!modes.has(m.mode))throw Error('Ungültige Nachricht im Import.');
      return {role:m.role,content:m.content,mode:m.mode,checkpoint:typeof m.checkpoint==='string'?m.checkpoint.slice(0,64):undefined,meta:'Importierter Verlauf · nicht neu berechnet'};
    });
    return {title:data.title.slice(0,80)||'Importierter Chat',messages,draft:'',temporary:false,imported:true};
  }
  const api={context,blocks,render,parseImport};
  if(typeof module!=='undefined')module.exports=api;else root.HeliosCore=api;
})(typeof window==='undefined'?{}:window);
