"""Deterministic Neral-owned behavior curriculum for the first Arctus 350M chatbot pilot.

No external dataset text is used here. Every training/validation example is generated from
project-authored templates and small factual primitives in this source file. The curriculum
teaches chat format, instruction obedience, uncertainty, concise answers, bilingual behavior
and multi-turn state. It is DEVELOPMENT data, not a held-out quality benchmark.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter

VOCAB_SIZE=16_384
MAX_CONTEXT=512
DATASET_SCHEMA="neral_chatbot_v1_project_owned_behavior_v1"

DE_EXPLANATIONS=(
    ("Warum fällt ein Gegenstand auf der Erde nach unten?","Weil die Gravitation ihn zur Erde beschleunigt."),
    ("Was beschreibt eine Umlaufbahn?","Sie beschreibt den Weg eines Körpers um einen anderen Körper unter dem Einfluss von Kräften."),
    ("Warum braucht ein Computer Arbeitsspeicher?","Arbeitsspeicher hält Daten kurzfristig bereit, damit der Prozessor schnell darauf zugreifen kann."),
    ("Was macht eine Variable in einem Programm?","Eine Variable speichert einen Wert, auf den das Programm später zugreifen oder den es verändern kann."),
    ("Warum ist Wasser bei Raumtemperatur flüssig?","Unter normalen Bedingungen liegt die Raumtemperatur zwischen Schmelz- und Siedepunkt von Wasser."),
    ("Was ist der Unterschied zwischen Masse und Gewichtskraft?","Die Masse beschreibt die Materiemenge, die Gewichtskraft ist die Gravitationskraft auf diese Masse."),
    ("Warum haben Flugzeuge Tragflächen?","Tragflächen erzeugen bei geeigneter Anströmung aerodynamischen Auftrieb."),
    ("Was ist eine Datenbank?","Eine Datenbank speichert strukturierte Daten so, dass sie gezielt gelesen und verändert werden können."),
    ("Was bedeutet Geschwindigkeit?","Geschwindigkeit beschreibt, wie stark sich der Ort eines Körpers pro Zeit ändert."),
    ("Was ist elektrische Spannung?","Elektrische Spannung beschreibt die Energieänderung pro Ladungsmenge zwischen zwei Punkten."),
)
EN_EXPLANATIONS=(
    ("Why does an object fall toward Earth?","Because gravity accelerates it toward Earth."),
    ("What does an orbit describe?","It describes the path of one body around another under the influence of forces."),
    ("Why does a computer need memory?","Memory keeps data available temporarily so the processor can access it quickly."),
    ("What does a variable do in a program?","A variable stores a value that a program can read or change later."),
    ("Why is water liquid at room temperature?","Under ordinary conditions, room temperature lies between water's melting and boiling points."),
    ("What is the difference between mass and weight?","Mass describes an amount of matter, while weight is the gravitational force acting on that mass."),
    ("Why do aircraft have wings?","Wings generate aerodynamic lift when air flows around them appropriately."),
    ("What is a database?","A database stores structured data so it can be queried and changed systematically."),
    ("What does speed describe?","Speed describes how much distance is covered per unit of time."),
    ("What is electric voltage?","Voltage describes energy change per unit charge between two points."),
)

DE_TRANSLATIONS=(
    ("Guten Morgen","Good morning"),("Bis später","See you later"),("Wie geht es dir?","How are you?"),
    ("Das ist eine gute Idee.","That is a good idea."),("Der Zug kommt gleich.","The train is arriving soon."),
    ("Ich lerne gerade Physik.","I am studying physics right now."),
)
EN_TRANSLATIONS=tuple((b,a) for a,b in DE_TRANSLATIONS)

IDENTITY_DE=(
    "Ich bin Helios, der lokale Chat-Assistent des Neral-Projekts.",
    "Ich bin Helios, ein lokaler KI-Assistent auf Basis des Neral-Modells.",
)
IDENTITY_EN=(
    "I am Helios, the local chat assistant of the Neral project.",
    "I am Helios, a local AI assistant based on the Neral model.",
)


def norm(text:str)->str:
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())


def text_sha(text:str)->str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def conversation(identifier:str,category:str,language:str,turns:list[tuple[str,str]])->dict:
    if language not in {"de","en"} or not turns:
        raise ValueError("invalid conversation metadata")
    clean=[]
    for user,assistant in turns:
        if not isinstance(user,str) or not user.strip() or not isinstance(assistant,str) or not assistant.strip():
            raise ValueError("empty turn")
        if any(marker in user or marker in assistant for marker in ("<|bos|>","<|eos|>","<|pad|>")):
            raise ValueError("special-token injection")
        clean.append({"user":user.strip(),"assistant":assistant.strip()})
    return {"id":identifier,"category":category,"language":language,"turns":clean}


def _range(split:str,count:int,offset:int=0):
    base=0 if split=="train" else 10_000
    return range(base+offset,base+offset+count)


def build_split(split:str)->list[dict]:
    if split not in {"train","validation"}:
        raise ValueError("split must be train or validation")
    scale=1 if split=="train" else 0
    rows=[]

    exact_count=80 if split=="train" else 16
    for i in _range(split,exact_count,100):
        token=("KUPFER" if i%2==0 else "ORBITA")+str(i)
        rows.append(conversation(f"{split}-exact-de-{i}","exact_instruction","de",
            [(f"Antworte nur mit {token}. Verwende kein weiteres Wort.",token)]))
        token_en=("COPPER" if i%2==0 else "ORBITA")+str(i)
        rows.append(conversation(f"{split}-exact-en-{i}","exact_instruction","en",
            [(f"Reply with only {token_en}. Do not add any other word.",token_en)]))

    arithmetic_count=90 if split=="train" else 18
    for i in _range(split,arithmetic_count,400):
        a=17+(i%43);b=6+((i*7)%29)
        rows.append(conversation(f"{split}-math-de-{i}","arithmetic","de",
            [(f"Berechne {a} + {b} und antworte in genau einem kurzen Satz.",f"Das Ergebnis ist {a+b}.")]))
        c=9+(i%17);d=3+((i*5)%11)
        rows.append(conversation(f"{split}-math-en-{i}","arithmetic","en",
            [(f"Calculate {c} × {d} and answer in exactly one short sentence.",f"The result is {c*d}.")]))

    memory_count=90 if split=="train" else 18
    for i in _range(split,memory_count,800):
        code=f"NX-{i:05d}"
        rows.append(conversation(f"{split}-memory-de-{i}","multi_turn_state","de",[
            (f"Merke dir für dieses Gespräch den Code {code}.","Okay, ich merke mir den Code für dieses Gespräch."),
            ("Welchen Code habe ich dir gerade genannt?",f"Du hast mir den Code {code} genannt."),
        ]))
        city=f"Velora-{i}"
        rows.append(conversation(f"{split}-memory-en-{i}","multi_turn_state","en",[
            (f"For this conversation, remember the fictional city name {city}.","Okay, I will remember that fictional city name for this conversation."),
            ("Which fictional city name did I just give you?",f"You gave me the name {city}."),
        ]))

    correction_count=45 if split=="train" else 9
    for i in _range(split,correction_count,1200):
        first=f"A-{i:04d}";second=f"B-{i:04d}"
        rows.append(conversation(f"{split}-correct-de-{i}","multi_turn_correction","de",[
            (f"Für diese Unterhaltung ist mein Kennwort {first}.","Verstanden."),
            (f"Korrektur: Das Kennwort ist jetzt {second}.","Verstanden, ich verwende jetzt das korrigierte Kennwort."),
            ("Wie lautet das aktuelle Kennwort?",f"Das aktuelle Kennwort lautet {second}."),
        ]))
        rows.append(conversation(f"{split}-correct-en-{i}","multi_turn_correction","en",[
            (f"For this conversation my code word is {first}.","Understood."),
            (f"Correction: the code word is now {second}.","Understood, I will use the corrected code word now."),
            ("What is the current code word?",f"The current code word is {second}."),
        ]))

    unknown_count=70 if split=="train" else 14
    for i in _range(split,unknown_count,1500):
        object_name=f"Zeran-{i}"
        rows.append(conversation(f"{split}-unknown-de-{i}","uncertainty","de",
            [(f"Wie viele Einwohner hat die erfundene Stadt {object_name} im Jahr 2047?",
              f"Das weiß ich nicht verlässlich. {object_name} ist hier als erfundene Stadt beschrieben, daher sollte ich keine Einwohnerzahl erfinden.")]))
        rows.append(conversation(f"{split}-unknown-en-{i}","uncertainty","en",
            [(f"What was the exact population of the fictional city {object_name} in 2047?",
              f"I cannot know that reliably. {object_name} is described here as fictional, so I should not invent a population figure.")]))    

    bullet_count=45 if split=="train" else 9
    topics_de=(("einen Lernplan",("Ein klares Ziel festlegen","Lernzeit in kleine Abschnitte teilen")),
               ("eine Reisevorbereitung",("Dokumente und Zeiten prüfen","Wichtige Dinge früh packen")),
               ("sauberen Code",("Aussagekräftige Namen verwenden","Funktionen klein und klar halten")))
    topics_en=(("a study plan",("Set a clear goal","Split study time into small sessions")),
               ("trip preparation",("Check documents and times","Pack important items early")),
               ("clean code",("Use meaningful names","Keep functions small and clear")))
    for idx,i in enumerate(_range(split,bullet_count,1800)):
        td,points=topics_de[idx%len(topics_de)]
        rows.append(conversation(f"{split}-bullets-de-{i}","format_control","de",
            [(f"Beispiel {i}: Nenne genau zwei kurze Punkte für {td}.",f"- {points[0]}\n- {points[1]}")]))
        te,points_en=topics_en[idx%len(topics_en)]
        rows.append(conversation(f"{split}-bullets-en-{i}","format_control","en",
            [(f"Example {i}: Give exactly two short points for {te}.",f"- {points_en[0]}\n- {points_en[1]}")]))

    clarify_count=35 if split=="train" else 7
    for i in _range(split,clarify_count,2100):
        rows.append(conversation(f"{split}-clarify-de-{i}","clarification","de",
            [(f"Plane das Projekt {i} für mich.","Welches Ziel, welchen Zeitraum und welche wichtigsten Einschränkungen soll ich für die Planung berücksichtigen?")]))
        rows.append(conversation(f"{split}-clarify-en-{i}","clarification","en",
            [(f"Plan project {i} for me.","What goal, timeframe, and main constraints should I use for the plan?")]))

    explanation_repeats=10 if split=="train" else 1
    de_pool=DE_EXPLANATIONS[:8] if split=="train" else DE_EXPLANATIONS[8:]
    en_pool=EN_EXPLANATIONS[:8] if split=="train" else EN_EXPLANATIONS[8:]
    for rep in range(explanation_repeats):
        for j,(q,a) in enumerate(de_pool):
            suffix=f" Formuliere die Antwort kurz. Variante {rep+1}." if split=="train" else " Antworte kurz."
            rows.append(conversation(f"{split}-explain-de-{rep}-{j}","short_explanation","de",[(q+suffix,a)]))
        for j,(q,a) in enumerate(en_pool):
            suffix=f" Keep the answer short. Variant {rep+1}." if split=="train" else " Keep the answer short."
            rows.append(conversation(f"{split}-explain-en-{rep}-{j}","short_explanation","en",[(q+suffix,a)]))

    translation_repeats=8 if split=="train" else 1
    de_translations=DE_TRANSLATIONS[:4] if split=="train" else DE_TRANSLATIONS[4:]
    en_translations=EN_TRANSLATIONS[:4] if split=="train" else EN_TRANSLATIONS[4:]
    for rep in range(translation_repeats):
        for j,(src,dst) in enumerate(de_translations):
            rows.append(conversation(f"{split}-translate-de-{rep}-{j}","bilingual","de",
                [(f"Variante {rep+1}: Übersetze ins Englische und gib nur die Übersetzung aus: {src}",dst)]))
        for j,(src,dst) in enumerate(en_translations):
            rows.append(conversation(f"{split}-translate-en-{rep}-{j}","bilingual","en",
                [(f"Version {rep+1}: Translate into German and output only the translation: {src}",dst)]))

    identity_count=24 if split=="train" else 6
    for idx,i in enumerate(_range(split,identity_count,2500)):
        rows.append(conversation(f"{split}-identity-de-{i}","identity","de",
            [(f"Wer bist du? Antworte knapp. Referenz {i}.",IDENTITY_DE[idx%len(IDENTITY_DE)])]))
        rows.append(conversation(f"{split}-identity-en-{i}","identity","en",
            [(f"Who are you? Answer briefly. Reference {i}.",IDENTITY_EN[idx%len(IDENTITY_EN)])]))

    ids=[row["id"] for row in rows]
    if len(ids)!=len(set(ids)):raise AssertionError("duplicate conversation id")
    prompts=[]
    for row in rows:
        prompts.extend(norm(turn["user"]) for turn in row["turns"])
    if len(prompts)!=len(set(prompts)):raise AssertionError("duplicate user prompt inside split")
    return rows


def encode_conversation(row:dict,tokenizer:object,bos:int,eos:int,seq:int=MAX_CONTEXT)->dict:
    if not (0<=bos<VOCAB_SIZE and 0<=eos<VOCAB_SIZE and bos!=eos):
        raise ValueError("invalid BOS/EOS")
    full=[bos];supervised=set()
    for index,turn in enumerate(row["turns"]):
        prefix=("User: " if index==0 else "\nUser: ")+turn["user"]+"\nAssistant:"
        prefix_ids=list(tokenizer.encode(prefix,add_special_tokens=False).ids)
        answer_ids=list(tokenizer.encode(" "+turn["assistant"],add_special_tokens=False).ids)
        if not answer_ids:raise ValueError("empty tokenized answer")
        full.extend(prefix_ids)
        start=len(full);full.extend(answer_ids)
        supervised.update(range(start,start+len(answer_ids)))
    full.append(eos);supervised.add(len(full)-1)
    if len(full)-1>seq:raise ValueError("conversation exceeds context; never truncate")
    if any(type(token) is not int or not 0<=token<VOCAB_SIZE for token in full):
        raise ValueError("out-of-range token")
    x=full[:-1]
    labels=[full[i+1] if i+1 in supervised else -100 for i in range(len(x))]
    if labels.count(eos)!=1 or labels[-1]!=eos:
        raise AssertionError("terminal EOS supervision broken")
    supervised_count=sum(label!=-100 for label in labels)
    return {"id":row["id"],"category":row["category"],"language":row["language"],
            "input_ids":x,"labels":labels,"supervised_tokens":supervised_count,
            "turn_count":len(row["turns"])}


def build_encoded(tokenizer:object,bos:int,eos:int,seq:int=MAX_CONTEXT)->tuple[dict[str,list[dict]],dict]:
    raw={split:build_split(split) for split in ("train","validation")}
    encoded={split:[encode_conversation(row,tokenizer,bos,eos,seq) for row in rows] for split,rows in raw.items()}
    train_prompts={norm(turn["user"]) for row in raw["train"] for turn in row["turns"]}
    valid_prompts={norm(turn["user"]) for row in raw["validation"] for turn in row["turns"]}
    if train_prompts & valid_prompts:raise AssertionError("train/validation prompt overlap")
    payload=json.dumps(raw,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
    report={
        "schema":DATASET_SCHEMA,
        "classification":"project_owned_deterministic_behavior_sft_development_only",
        "external_text_sources":False,
        "human_private_data":False,
        "train_rows":len(encoded["train"]),
        "validation_rows":len(encoded["validation"]),
        "train_supervised_tokens":sum(r["supervised_tokens"] for r in encoded["train"]),
        "validation_supervised_tokens":sum(r["supervised_tokens"] for r in encoded["validation"]),
        "category_counts":{
            split:dict(sorted(Counter(row["category"] for row in raw[split]).items()))
            for split in raw
        },
        "language_counts":{
            split:dict(sorted(Counter(row["language"] for row in raw[split]).items()))
            for split in raw
        },
        "max_encoded_tokens":max(len(r["input_ids"]) for rows in encoded.values() for r in rows),
        "dataset_sha256":hashlib.sha256(payload).hexdigest(),
        "quality_promotion":False,
    }
    return encoded,report


def prompt_fingerprints(rows:list[dict])->set[str]:
    out=set()
    for row in rows:
        for turn in row["turns"]:
            out.add(text_sha(norm(turn["user"])))
            out.add(text_sha(norm("User: "+turn["user"]+"\nAssistant:")))
    return out
