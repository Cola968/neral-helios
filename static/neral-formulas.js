"use strict";
/* NERAL formula library: offline, vetted examples and zero model execution.
 * Formula cards are inert text rendered by the locally bundled KaTeX only.
 */
(() => {
  const raw=[
    ["binom2","Algebra","1. binomische Formel","(a+b)^2=a^2+2ab+b^2","Quadrat einer Summe"],
    ["binom3","Algebra","2. binomische Formel","(a-b)^2=a^2-2ab+b^2","Quadrat einer Differenz"],
    ["quadratic","Algebra","Mitternachtsformel","x_{1,2}=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}","Quadratische Gleichungen mit a ungleich null"],
    ["power","Algebra","Potenzgesetze","a^m\\cdot a^n=a^{m+n}","Gleiche Basis bei der Multiplikation"],
    ["logarithm","Algebra","Logarithmusgesetz","\\ln(ab)=\\ln a+\\ln b","Für positive a und b"],
    ["pythagoras","Algebra","Satz des Pythagoras","a^2+b^2=c^2","Nur im rechtwinkligen Dreieck"],
    ["derivative","Analysis","Potenzregel","\\frac{d}{dx}x^n=nx^{n-1}","Ableitung einer Potenz"],
    ["chain","Analysis","Kettenregel","(f\\circ g)'(x)=f'(g(x))\\,g'(x)","Ableitung verketteter Funktionen"],
    ["product-rule","Analysis","Produktregel","(fg)'=f'g+fg'","Produkt zweier differenzierbarer Funktionen"],
    ["quotient","Analysis","Quotientenregel","\\left(\\frac fg\\right)'=\\frac{f'g-fg'}{g^2}","An Stellen mit g ungleich null"],
    ["integral","Analysis","Potenz integrieren","\\int x^n\\,dx=\\frac{x^{n+1}}{n+1}+C","Gilt für n ungleich −1"],
    ["fundamental","Analysis","Hauptsatz der Analysis","\\int_a^b f(x)\\,dx=F(b)-F(a)","F ist eine Stammfunktion von f"],
    ["gradient","Analysis","Gradient","\\nabla f=\\left(\\frac{\\partial f}{\\partial x},\\frac{\\partial f}{\\partial y},\\frac{\\partial f}{\\partial z}\\right)","Räumliche Änderungsraten eines Skalarfelds"],
    ["matrix","Analysis","Determinante 2 × 2","\\det\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}=ad-bc","Determinante einer 2×2-Matrix"],
    ["newton","Mechanik","Zweites Newtonsches Gesetz","\\vec F=m\\vec a","Resultierende Kraft bei konstanter Masse"],
    ["kinetic","Mechanik","Kinetische Energie","E_{\\mathrm{kin}}=\\frac12mv^2","Nichtrelativistische Bewegungsenergie"],
    ["momentum","Mechanik","Impuls","\\vec p=m\\vec v","Impuls bei konstanter Masse"],
    ["gravitation","Mechanik","Newtonsches Gravitationsgesetz","F=G\\frac{m_1m_2}{r^2}","Gravitationskraft zweier Punktmassen"],
    ["escape","Mechanik","Fluchtgeschwindigkeit","v_{\\mathrm{esc}}=\\sqrt{\\frac{2GM}{r}}","Ohne Atmosphäre und weitere Körper"],
    ["spring","Mechanik","Federpendel","T=2\\pi\\sqrt{\\frac{m}{k}}","Idealisierte ungedämpfte Feder"],
    ["wave","Mechanik","Wellengeschwindigkeit","v=f\\lambda","Frequenz und Wellenlänge"],
    ["pressure","Mechanik","Druck","p=\\frac FA","Kraft senkrecht zur Fläche"],
    ["mass-energy","Relativität","Masse-Energie-Äquivalenz","E_0=mc^2","Ruheenergie eines Körpers"],
    ["gamma","Relativität","Lorentzfaktor","\\gamma=\\frac{1}{\\sqrt{1-v^2/c^2}}","Nur für Geschwindigkeiten kleiner als c"],
    ["time-dilation","Relativität","Zeitdilatation","\\Delta t=\\gamma\\Delta\\tau","Eigenzeit und Zeit im Bezugssystem"],
    ["length","Relativität","Längenkontraktion","L=\\frac{L_0}{\\gamma}","Länge in Bewegungsrichtung"],
    ["einstein","Relativität","Einsteinsche Feldgleichungen","G_{\\mu\\nu}+\\Lambda g_{\\mu\\nu}=\\frac{8\\pi G}{c^4}T_{\\mu\\nu}","Allgemeine Relativitätstheorie"],
    ["schwarzschild","Relativität","Schwarzschild-Radius","r_s=\\frac{2GM}{c^2}","Schwarzschild-Lösung für nichtrotierende Masse"],
    ["coulomb","Elektromagnetismus","Coulomb-Gesetz","F=\\frac{1}{4\\pi\\varepsilon_0}\\frac{|q_1q_2|}{r^2}","Betrag der elektrostatischen Kraft im Vakuum"],
    ["ohm","Elektromagnetismus","Ohmsches Gesetz","U=RI","Für ohmsche Widerstände"],
    ["electric-power","Elektromagnetismus","Elektrische Leistung","P=UI","Gleichstrom bzw. Momentanwerte"],
    ["capacitance","Elektromagnetismus","Kapazität","C=\\frac QU","Ladung je Spannung"],
    ["induction","Elektromagnetismus","Faradaysches Induktionsgesetz","U_{\\mathrm{ind}}=-\\frac{d\\Phi_B}{dt}","Vorzeichen gemäß Lenzscher Regel"],
    ["gauss","Elektromagnetismus","Gaußsches Gesetz","\\nabla\\cdot\\vec E=\\frac{\\rho}{\\varepsilon_0}","Maxwell-Gleichung im Vakuum"],
    ["photon","Elektromagnetismus","Photonenenergie","E=hf","Energie eines Photons"],
    ["kepler","Astronomie","Drittes Keplersches Gesetz","\\frac{T^2}{a^3}=\\frac{4\\pi^2}{G(M+m)}","Zweikörpersystem: große Halbachse a"],
    ["luminosity","Astronomie","Strahlungsfluss","F=\\frac{L}{4\\pi d^2}","Isotrope Abstrahlung ohne Absorption"],
    ["distance-modulus","Astronomie","Entfernungsmodul","m-M=5\\log_{10}\\left(\\frac{d}{10\\,\\mathrm{pc}}\\right)","Ohne interstellare Extinktion"],
    ["hubble","Astronomie","Hubble-Lemaître-Gesetz","v\\approx H_0d","Näherung für kleine Rotverschiebungen"],
    ["stefan","Astronomie","Stefan-Boltzmann-Gesetz","L=4\\pi R^2\\sigma T^4","Idealisierter Schwarzkörper"],
    ["redshift","Astronomie","Rotverschiebung","1+z=\\frac{\\lambda_{\\mathrm{obs}}}{\\lambda_{\\mathrm{emit}}}","Vergleich beobachteter und emittierter Wellenlänge"],
    ["water","Chemie","Wasserbildung","\\ce{2H2 + O2 -> 2H2O}","Ausgeglichene chemische Reaktionsgleichung"],
    ["neutralisation","Chemie","Neutralisation","\\ce{H+ + OH- -> H2O}","Vereinfachte ionische Darstellung"],
    ["salt","Chemie","Salzdissoziation","\\ce{NaCl -> Na+ + Cl-}","Dissoziation in wässriger Lösung"],
    ["acid","Chemie","Säuregleichgewicht","\\ce{HA <=> H+ + A-}","Allgemeine Gleichgewichtsreaktion"],
    ["ph","Chemie","pH-Wert","\\mathrm{pH}=-\\log_{10}(a_{\\ce{H+}})","Über die dimensionslose Wasserstoffionenaktivität"],
    ["ideal-gas","Chemie","Ideale Gasgleichung","pV=nRT","Näherung für ideale Gase"],
    ["probability","Statistik","Bedingte Wahrscheinlichkeit","P(A\\mid B)=\\frac{P(A\\cap B)}{P(B)}","Nur wenn P(B) größer als null"],
    ["binomial","Statistik","Binomialverteilung","P(X=k)=\\binom nk p^k(1-p)^{n-k}","n unabhängige Bernoulli-Versuche"],
    ["mean","Statistik","Arithmetisches Mittel","\\bar x=\\frac1n\\sum_{i=1}^n x_i","Mittelwert von n Beobachtungen"],
    ["normal","Statistik","Normalverteilung","f(x)=\\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}","Dichtefunktion mit Standardabweichung σ > 0"]
  ];
  const catalog=raw.map(([id,category,name,latex,note])=>
    Object.freeze({id,category,name,latex,note}));
  Object.freeze(catalog);
  const categories=Object.freeze([...new Set(catalog.map(x=>x.category))]);
  const normalize=text=>String(text||"").normalize("NFD")
    .replace(/[\u0300-\u036f]/g,"").toLocaleLowerCase("de-DE");
  function search(query="",category="Alle"){
    const q=normalize(query.trim());
    return catalog.filter(x=>(category==="Alle"||x.category===category)&&
      (!q||normalize([x.id,x.category,x.name,x.note].join(" ")).includes(q)));
  }
  function insertText(original,start,end,snippet,maxLength=4000){
    if(typeof original!=="string"||typeof snippet!=="string"||
       !Number.isInteger(start)||!Number.isInteger(end)||
       start<0||start>end||end>original.length||snippet.length>800)return null;
    const value=original.slice(0,start)+snippet+original.slice(end);
    if(value.length>maxLength)return null;
    return {text:value,caret:start+snippet.length};
  }
  const $=id=>document.getElementById(id);
  const picker=$("formulaLibraryPanel"),toggle=$("formulaLibraryToggle"),
        list=$("formulaLibraryList"),input=$("formulaLibrarySearch"),
        category=$("formulaLibraryCategory"),prompt=$("prompt"),
        count=$("formulaLibraryCount"),status=$("formulaLibraryStatus");
  const api={catalog,categories,search,insertText};
  window.NeralFormulas=api;
  if(!picker||!toggle||!list||!input||!category||!prompt)return;
  const create=(tag,cls,text)=>{
    const el=document.createElement(tag);
    if(cls)el.className=cls;
    if(text!==undefined)el.textContent=String(text);
    return el;
  };
  for(const name of ["Alle",...categories]){
    const option=create("option","",name);option.value=name;category.append(option);
  }
  function close(){
    picker.hidden=true;toggle.setAttribute("aria-expanded","false");
  }
  function open(){
    picker.hidden=false;toggle.setAttribute("aria-expanded","true");
    draw();input.focus();
  }
  function insertFormula(item){
    const markup="$"+item.latex+"$";
    const beginning=Number.isInteger(prompt.selectionStart)?prompt.selectionStart:prompt.value.length;
    const ending=Number.isInteger(prompt.selectionEnd)?prompt.selectionEnd:beginning;
    const padded=(beginning>0&&!/\s$/.test(prompt.value.slice(0,beginning))?" ":"")+
      markup+(ending<prompt.value.length&&!/^\s/.test(prompt.value.slice(ending))?" ":"");
    const next=insertText(prompt.value,beginning,ending,padded,
      Math.min(4000,prompt.maxLength>0?prompt.maxLength:4000));
    if(!next){if(status)status.textContent="Nachricht zu lang für diese Formel.";return;}
    if(typeof prompt.setRangeText==="function")
      prompt.setRangeText(padded,beginning,ending,"end");
    else {
      prompt.value=next.text;
      prompt.selectionStart=next.caret;prompt.selectionEnd=next.caret;
    }
    prompt.dispatchEvent(new Event("input",{bubbles:true}));
    close();prompt.focus();
    if(status)status.textContent=item.name+" eingefügt. Vor dem Senden prüfen.";
  }
  function draw(){
    const matches=search(input.value,category.value||"Alle");
    const cards=matches.map(item=>{
      const card=create("article","neral-formula-card");
      const text=create("div","neral-formula-description");
      text.append(create("strong","",item.name),create("small","",item.note));
      const formula=create("div","neral-formula-preview","\\["+item.latex+"\\]");
      window.NeralMath?.render(formula);
      const insert=create("button","neral-formula-insert","Einfügen");
      insert.type="button";insert.setAttribute("aria-label",item.name+" einfügen");
      insert.addEventListener("click",()=>insertFormula(item));
      card.append(text,formula,insert);return card;
    });
    if(!cards.length)
      cards.push(create("p","neral-formula-empty","Keine passenden Formeln gefunden."));
    list.replaceChildren(...cards);
    if(count)count.textContent=matches.length+" Formeln";
  }
  toggle.addEventListener("click",()=>picker.hidden?open():close());
  input.addEventListener("input",draw);
  category.addEventListener("change",draw);
  picker.addEventListener("keydown",event=>{
    if(event.key==="Escape"){event.preventDefault();close();toggle.focus();}
  });
  document.addEventListener("keydown",event=>{
    if(event.key==="Escape"&&!picker.hidden){close();toggle.focus();}
  });
  draw();
})();
