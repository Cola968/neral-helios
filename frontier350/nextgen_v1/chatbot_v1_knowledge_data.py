"""Project-owned knowledge/truthfulness curriculum for Chatbot-v1.1 repair.

Stable factual primitives are authored in this file. No external corpus text, private user
content, public benchmark answer, or sealed holdout is used. Validation uses different
wording and identifiers; it is training-development validation, not an independent holdout.
"""
from __future__ import annotations

import hashlib,json
from collections import Counter

from chatbot_v1_data import conversation,encode_conversation,norm,prompt_fingerprints

SCHEMA="neral_chatbot_v1_knowledge_truthfulness_v1"

FACTS=(
 ("science","Die Erde bewegt sich auf einer Umlaufbahn um die Sonne.","Die Erde umkreist die Sonne.",
  "Earth moves in an orbit around the Sun.","Earth orbits the Sun."),
 ("science","Bei welcher Temperatur gefriert reines Wasser ungefähr unter normalem Luftdruck?","Bei ungefähr 0 °C.",
  "At about what temperature does pure water freeze at normal atmospheric pressure?","At about 0 °C."),
 ("science","Bei welcher Temperatur siedet reines Wasser ungefähr bei normalem Luftdruck auf Meereshöhe?","Bei ungefähr 100 °C.",
  "At about what temperature does pure water boil at normal atmospheric pressure near sea level?","At about 100 °C."),
 ("science","Wie groß ist die Fallbeschleunigung nahe der Erdoberfläche ungefähr?","Ungefähr 9,81 m/s².",
  "What is the approximate gravitational acceleration near Earth's surface?","About 9.81 m/s²."),
 ("science","Wie schnell ist Licht im Vakuum?","Exakt 299.792.458 m/s.",
  "How fast does light travel in vacuum?","Exactly 299,792,458 m/s."),
 ("science","Welche elektrische Ladung hat ein Elektron?","Ein Elektron ist negativ geladen.",
  "What electric charge does an electron have?","An electron has negative charge."),
 ("science","Welche elektrische Ladung hat ein Proton?","Ein Proton ist positiv geladen.",
  "What electric charge does a proton have?","A proton has positive charge."),
 ("science","Was transportiert DNA in Lebewesen?","DNA trägt genetische Information.",
  "What does DNA carry in living organisms?","DNA carries genetic information."),
 ("science","Was ist der Mond in Bezug auf die Erde?","Der Mond ist der natürliche Satellit der Erde.",
  "What is the Moon relative to Earth?","The Moon is Earth's natural satellite."),
 ("science","Welcher Planet ist der größte im Sonnensystem?","Jupiter ist der größte Planet im Sonnensystem.",
  "Which planet is the largest in the Solar System?","Jupiter is the largest planet in the Solar System."),
 ("science","Kann sich Schall durch ein ideales Vakuum ausbreiten?","Nein. Schall benötigt ein Medium.",
  "Can sound travel through an ideal vacuum?","No. Sound requires a medium."),
 ("science","Wie berechnet man die Dichte eines Stoffes?","Dichte ist Masse geteilt durch Volumen.",
  "How is density calculated?","Density is mass divided by volume."),
 ("science","Kann Verdunstung unterhalb des Siedepunkts stattfinden?","Ja. Verdunstung kann unterhalb des Siedepunkts stattfinden.",
  "Can evaporation occur below the boiling point?","Yes. Evaporation can occur below the boiling point."),
 ("computing","Ist eine Python-Liste veränderbar?","Ja. Eine Python-Liste ist veränderbar.",
  "Is a Python list mutable?","Yes. A Python list is mutable."),
 ("computing","Ist ein Python-Tupel normalerweise veränderbar?","Nein. Ein Python-Tupel ist unveränderlich.",
  "Is a Python tuple normally mutable?","No. A Python tuple is immutable."),
 ("computing","Wofür wird RAM in einem Computer genutzt?","RAM hält aktuell benötigte Daten und Programme kurzfristig bereit.",
  "What is RAM used for in a computer?","RAM temporarily holds data and programs that are currently needed."),
 ("computing","Was macht eine CPU grundsätzlich?","Eine CPU führt Programminstruktionen aus.",
  "What does a CPU fundamentally do?","A CPU executes program instructions."),
 ("computing","Was ist HTTP?","HTTP ist ein Anwendungsprotokoll zur Übertragung von Web-Ressourcen.",
  "What is HTTP?","HTTP is an application protocol used to transfer web resources."),
 ("computing","Ist TCP verbindungsorientiert?","Ja. TCP ist verbindungsorientiert.",
  "Is TCP connection-oriented?","Yes. TCP is connection-oriented."),
 ("computing","Ist UDP verbindungsorientiert?","Nein. UDP ist verbindungslos.",
  "Is UDP connection-oriented?","No. UDP is connectionless."),
 ("computing","Wozu dient ein Primärschlüssel in einer relationalen Tabelle?","Er identifiziert einen Datensatz eindeutig.",
  "What is a primary key used for in a relational table?","It uniquely identifies a record."),
 ("computing","Welche Ziffern verwendet das Binärsystem?","0 und 1.",
  "Which digits does the binary system use?","0 and 1."),
 ("computing","Was speichert ein Git-Commit grundsätzlich?","Er hält einen versionierten Zustand samt Metadaten und Verweisen fest.",
  "What does a Git commit fundamentally record?","It records a versioned state with metadata and references."),
 ("math","Wie viele Meter sind ein Kilometer?","1.000 Meter.",
  "How many meters are in one kilometer?","1,000 meters."),
 ("math","Wie groß ist die Winkelsumme eines Dreiecks in der euklidischen Ebene?","180 Grad.",
  "What is the angle sum of a triangle in Euclidean geometry?","180 degrees."),
 ("math","Wie lautet der Umfang eines Kreises mit Radius r?","2 · π · r.",
  "What is the circumference of a circle with radius r?","2 · π · r."),
 ("math","Wie lautet die Fläche eines Quadrats mit Seitenlänge a?","a².",
  "What is the area of a square with side length a?","a²."),
 ("math","Was ist eine Primzahl?","Eine Primzahl ist eine natürliche Zahl größer als 1 mit genau zwei positiven Teilern.",
  "What is a prime number?","A prime number is an integer greater than 1 with exactly two positive divisors."),
 ("aviation","Wie wirkt aerodynamischer Widerstand relativ zur Bewegung durch die Luft?","Er wirkt der relativen Bewegung durch die Luft entgegen.",
  "How does aerodynamic drag act relative to motion through the air?","It acts opposite the relative motion through the air."),
 ("aviation","Warum können Groundspeed und Airspeed eines Flugzeugs unterschiedlich sein?","Wind verändert die Geschwindigkeit über Grund relativ zur Geschwindigkeit durch die Luft.",
  "Why can an aircraft's groundspeed and airspeed differ?","Wind changes speed over the ground relative to speed through the air."),
 ("transport","Wozu dient eine Eisenbahnweiche?","Sie ermöglicht Fahrzeugen den Wechsel zwischen unterschiedlichen Gleiswegen.",
  "What is a railway switch used for?","It lets rail vehicles move between different track routes."),
 ("transport","Was beschreibt Geschwindigkeit allgemein?","Sie beschreibt die Ortsänderung pro Zeit.",
  "What does velocity generally describe?","It describes change of position per unit time."),
)

DE_TEMPLATES=(
 "{q}",
 "Antworte kurz und sachlich: {q}",
 "Gib eine knappe, verlässliche Antwort. {q}",
)
EN_TEMPLATES=(
 "{q}",
 "Answer briefly and factually: {q}",
 "Give a concise, reliable answer. {q}",
)


def _knowledge_rows(split:str)->list[dict]:
    rows=[]
    if split=="train":
        templates_de=DE_TEMPLATES;templates_en=EN_TEMPLATES
    else:
        templates_de=("Prüffrage, kurz beantworten: {q}",)
        templates_en=("Validation question, answer briefly: {q}",)
    for i,(domain,dq,da,eq,ea) in enumerate(FACTS):
        for variant,template in enumerate(templates_de):
            rows.append(conversation(f"{split}-fact-de-{i}-{variant}","knowledge_"+domain,"de",
                                     [(template.format(q=dq),da)]))
        for variant,template in enumerate(templates_en):
            rows.append(conversation(f"{split}-fact-en-{i}-{variant}","knowledge_"+domain,"en",
                                     [(template.format(q=eq),ea)]))
    return rows


def _uncertainty_rows(split:str)->list[dict]:
    count=60 if split=="train" else 12
    base=0 if split=="train" else 10_000
    rows=[]
    for j in range(count):
        i=base+j
        name=f"Orvex-{i}"
        rows.append(conversation(f"{split}-unknown-de-{i}","truthfulness_unknown","de",[
            (f"Die erfundene Raumstation {name} wurde genannt, aber es gibt keine Angabe zu ihrer Masse. Wie groß ist ihre exakte Masse?",
             f"Das lässt sich aus den gegebenen Informationen nicht bestimmen. Für {name} wurde keine Masse angegeben.")
        ]))
        rows.append(conversation(f"{split}-unknown-en-{i}","truthfulness_unknown","en",[
            (f"The fictional research station {name} was named, but no population was given. What is its exact population?",
             f"It cannot be determined from the given information. No population was provided for {name}.")
        ]))
    return rows


def _dialogue_rows(split:str)->list[dict]:
    count=60 if split=="train" else 12
    base=20_000 if split=="train" else 30_000
    rows=[]
    for j in range(count):
        i=base+j
        old=f"N-{i}";new=f"R-{i}"
        rows.append(conversation(f"{split}-dialogue-de-{i}","dialogue_replay","de",[
            (f"Merke dir für dieses Gespräch die Kennung {old}.","Verstanden, ich merke mir die Kennung."),
            (f"Korrektur: Die Kennung lautet jetzt {new}.","Verstanden, ich verwende ab jetzt die korrigierte Kennung."),
            ("Welche Kennung gilt jetzt?",f"Jetzt gilt {new}."),
        ]))
        rows.append(conversation(f"{split}-dialogue-en-{i}","dialogue_replay","en",[
            (f"Remember the label {old} for this conversation.","Understood, I will remember the label."),
            (f"Correction: the label is now {new}.","Understood, I will use the corrected label now."),
            ("Which label applies now?",f"The current label is {new}."),
        ]))
    return rows


def _behavior_replay(split:str)->list[dict]:
    if split!="train":return []
    from chatbot_v1_data import build_split
    wanted={"exact_instruction","format_control","identity","multi_turn_state"}
    source=[row for row in build_split("train") if row["category"] in wanted]
    # Deterministic bounded replay: enough to protect behavior without dominating knowledge.
    return source[::3][:160]


def build_split(split:str)->list[dict]:
    if split not in {"train","validation"}:raise ValueError("invalid split")
    rows=_knowledge_rows(split)+_uncertainty_rows(split)+_dialogue_rows(split)+_behavior_replay(split)
    ids=[r["id"] for r in rows]
    if len(ids)!=len(set(ids)):raise AssertionError("duplicate row id")
    return rows


def build_encoded(tokenizer,bos:int,eos:int,seq:int=512)->tuple[dict[str,list[dict]],dict]:
    raw={s:build_split(s) for s in ("train","validation")}
    encoded={s:[encode_conversation(r,tokenizer,bos,eos,seq) for r in rows] for s,rows in raw.items()}
    payload=json.dumps(raw,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    return encoded,{
      "schema":SCHEMA,"classification":"project_owned_knowledge_truthfulness_dialogue_repair",
      "external_text_sources":False,"human_private_data":False,
      "train_rows":len(encoded["train"]),"validation_rows":len(encoded["validation"]),
      "train_supervised_tokens":sum(x["supervised_tokens"] for x in encoded["train"]),
      "validation_supervised_tokens":sum(x["supervised_tokens"] for x in encoded["validation"]),
      "category_counts":{s:dict(sorted(Counter(r["category"] for r in raw[s]).items())) for s in raw},
      "dataset_sha256":hashlib.sha256(payload).hexdigest(),"quality_promotion":False,
    }


def exact_screen_fingerprints()->set[str]:
    rows=build_split("train")+build_split("validation")
    return prompt_fingerprints(rows)
