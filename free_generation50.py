"""Frozen free generation diagnostics; never correct or sanitize raw model output."""
import argparse,ast,collections,hashlib,json,re,time
from pathlib import Path
import torch
from experiment import load_model,sha,amp,TOKENIZER

def prepare():
    out=Path('benchmarks50');out.mkdir(exist_ok=True)
    if (out/'regression.jsonl').exists():raise ValueError('Frozen suite already exists')
    rows=[]
    de=['Hallo, wie geht es dir?','Wer bist du?','Was kannst du?','Wie heißt du?','Antworte in einem Satz: Was ist Wasser?','Nenne drei Tiere.','Schreibe einen kurzen Satz über Berlin.','Was bedeutet schnell?','Ich verstehe das nicht. Bitte frage nach, was ich meine.','Antworte bitte auf Deutsch.']
    en=['Hello, how are you?','Who are you?','What can you do?','What is your name?','In one sentence: what is water?','Name three animals.','Write a short sentence about London.','What does fast mean?','I do not understand. Ask what I need explained.','Please reply in English.']
    for lang,qs in [('de',de),('en',en)]:
        for p in qs:rows.append({'language':lang,'domain':'chat','prompt':p})
    topics_de=['Gravitation','Photosynthese','ein Schwarzes Loch','ein Flughafen','eine Variable in Python','ein Regenbogen','ein Bruch','ein Algorithmus','ein Magnet','ein Vulkan','ein sicheres Passwort','ein Backup','Phishing-Erkennung','eine Schleife','eine Liste','eine Datenbank','ein Planet','ein Stromkreis','Reibung','Verdunstung']
    topics_en=['gravity','photosynthesis','a black hole','an airport','a Python variable','a rainbow','a fraction','an algorithm','a magnet','a volcano','a strong password','a backup','phishing detection','a loop','a list','a database','a planet','an electric circuit','friction','evaporation']
    for lang,topics in [('de',topics_de),('en',topics_en)]:
        for topic in topics:rows.append({'language':lang,'domain':'explanation','prompt':('Erkläre kurz und einfach: ' if lang=='de' else 'Explain briefly and simply: ')+topic})
    for lang in ['de','en']:
        for a,b in [(9,2),(17,8),(31,12),(7,6),(100,25)]:rows.append({'language':lang,'domain':'math','prompt':('Was ist ' if lang=='de' else 'What is ')+f'{a} + {b}?','expected':str(a+b)})
    for lang,qs in [('de',['Schreibe eine Python-Funktion, die zwei Zahlen addiert.','Erkläre den Fehler: print("Hallo"','Erkläre eine for-Schleife.','Was ist der Unterschied zwischen Liste und Tuple?','Wie prüft man eine Funktion mit einem Test?']),('en',['Write a Python function that adds two numbers.','Explain this error: print("Hello"','Explain a for loop.','How do a list and a tuple differ?','How do you test a function?'])]:
        for p in qs:rows.append({'language':lang,'domain':'coding','prompt':p})
    extra=['Fasse zusammen: Mia fährt mit dem Zug. Ihr Fahrrad bleibt zu Hause.','Verbessere den Satz: Ich habe gestern gehen Kino.','Formuliere höflicher: Gib mir das Buch!','Übersetze ins Englische: Das Fenster ist offen.','Übersetze ins Deutsche: The train is late.','Erkläre es einfacher. Vorherige Erklärung: Ein Algorithmus ist eine endliche Folge klarer Schritte.','Hlao, kanst du mir helfn?','Äpfel, Öl und süß: Schreibe diese drei Wörter ab.','Schreibe einen freundlichen Gruß mit einem Emoji.',' ','Summarize: Sam walks to school. It takes ten minutes.','Correct this sentence: He go to school yesterday.','Make this polite: Give me the book!','Translate into German: The window is open.','Translate into English: Der Zug kommt später.','Explain more simply: An algorithm is a finite sequence of instructions.','Helo, cn yu help me?','Repeat exactly: naïve café.','Write a friendly greeting with an emoji.','Reply briefly: '+('Please be clear. '*80)]
    for i,p in enumerate(extra):rows.append({'language':'de' if i<10 else 'en','domain':'robustness','prompt':p})
    assert len(rows)==100
    holdout=[{'language':lang,'domain':'heldout_explanation','prompt':('Beschreibe in zwei Sätzen: ' if lang=='de' else 'Describe in two sentences: ')+t} for lang,topics in [('de',['Trägheit','Dichte','eine Karte','ein Kompass','ein Mikroskop','eine Batterie','ein Fahrrad','ein Kalender','eine Bibliothek','ein Wörterbuch']),('en',['inertia','density','a map','a compass','a microscope','a battery','a bicycle','a calendar','a library','a dictionary'])] for t in topics]
    for split,rs in [('regression',rows),('holdout',holdout)]:
        for i,r in enumerate(rs):r['id']=f'{split}/{i:03d}'
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf8')
    (out/'manifest.json').write_text(json.dumps({'regression_prompts':100,'holdout_prompts':20,'hashes':{s:sha(out/(s+'.jsonl')) for s in ['regression','holdout']},'rules':'Frozen before pretraining pilot; heldout prompts are not executed in this pilot. No benchmark prompts added as SFT examples.','manual_rubric':'0 unintelligible; 1 words without usable statement; 2 partly understandable; 3 understandable and mostly relevant; 4 clear correct natural. Human/agent judgement is recorded separately, never inferred from loss.'},indent=2))
def diagnostics(ids,text):
    words=re.findall(r'\w+',text.casefold());repetition={}
    for n in range(1,5):
        grams=[tuple(words[i:i+n]) for i in range(len(words)-n+1)];repetition[str(n)]=sum(v-1 for v in collections.Counter(grams).values())/max(1,len(grams))
    raw=bytes(i-4 for i in ids if 4<=i<260)
    try:raw.decode('utf8');valid=True
    except UnicodeDecodeError:valid=False
    longest=max((sum(1 for _ in g) for _,g in __import__('itertools').groupby(ids)),default=0)
    de=len(set(words)&{'der','die','das','ist','und','ich','ein','eine','nicht','mit'});en=len(set(words)&{'the','is','and','i','a','an','not','with','you','to'})
    language='unknown' if max(de,en)<2 else 'de' if de>en else 'en' if en>de else 'mixed'
    return {'valid_utf8':valid,'output_bytes':len(raw),'characters':len(text),'repeated_word_ngrams':repetition,'longest_identical_token_run':longest,'language_heuristic':language,'eos':bool(ids and ids[-1]==2),'special_non_eos_tokens':sum(i in [0,1,3] for i in ids)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--checkpoint');ap.add_argument('--out');ap.add_argument('--max-tokens',type=int,default=64);args=ap.parse_args()
    if args.prepare:prepare();return
    torch.set_num_threads(4);torch.manual_seed(50013);torch.set_float32_matmul_precision('high');device=torch.device('cuda');m,p=load_model(args.checkpoint,device);m.eval();rows=[json.loads(l) for l in Path('benchmarks50/regression.jsonl').read_text(encoding='utf8').splitlines()];report={'checkpoint_sha256':sha(args.checkpoint),'parameters':m.parameter_report(),'suite_sha256':sha('benchmarks50/regression.jsonl'),'generation':{'temperature':0,'max_new_tokens':args.max_tokens,'seed':50013},'responses':[],'language_note':'Small stop-word heuristic; unknown does not mean correct language. No automated comprehensibility claim.'}
    for i,r in enumerate(rows):
        ids=TOKENIZER.encode('User: '+r['prompt']+'\nAssistant:',bos=True);x=torch.tensor([ids],device=device);torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode(),amp(device):out=m.generate(x,max_new_tokens=args.max_tokens,temperature=0,eos_id=2)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;generated=out[0,len(ids):].tolist();text=TOKENIZER.decode(generated)
        row=r|{'raw_output':text,'raw_token_ids':generated,'seconds':elapsed,'tokens_per_second':len(generated)/elapsed,'diagnostics':diagnostics(generated,text)}
        if 'expected' in r:row['numeric_exact']=text.strip().rstrip('.')==r['expected']
        report['responses'].append(row)
        if (i+1)%20==0:print(json.dumps({'completed':i+1}),flush=True)
    rs=report['responses'];report['summary']={'responses':len(rs),'valid_utf8_fraction':sum(r['diagnostics']['valid_utf8'] for r in rs)/len(rs),'eos_fraction':sum(r['diagnostics']['eos'] for r in rs)/len(rs),'mean_repeated_trigram_fraction':sum(r['diagnostics']['repeated_word_ngrams']['3'] for r in rs)/len(rs),'numeric_exact':sum(r.get('numeric_exact',False) for r in rs),'numeric_total':sum('numeric_exact' in r for r in rs),'mean_tokens_per_second':sum(r['tokens_per_second'] for r in rs)/len(rs),'mean_seconds':sum(r['seconds'] for r in rs)/len(rs)}
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report['summary']),flush=True)
if __name__=='__main__':main()
