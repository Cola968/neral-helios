# NERAL ORIGINAL MODEL FIRST — verbindliche Architekturentscheidung
Stand: 28. Oktober 2026 · Ziel: NERAL mit eigener Modellgewichts-Herkunft

## Entscheidung

**NERAL soll kein umbenanntes oder bloß per LoRA angepasstes Qwen sein.**
Der langfristige NERAL-Kern heißt **Arctus** und verwendet eine
unabhängige Modellarchitektur sowie von einem zufälligen Startpunkt
selbst trainierte Modellgewichte.

**Keine Qwen- oder sonstigen externen vortrainierten Gewichte, Embeddings,
LoRA-Adapter, Optimizerzustände oder Weight-Initialisierungen** in der
Arctus-Grundmodell-Linie. Ein LoRA-Fine-Tuning von Qwen bleibt in jedem Fall
eine abgeleitete Qwen-Modellvariante und erfüllt dieses Ziel **nicht**.

Qwen3.5 darf als deutlich gekennzeichneter, vorläufiger Produkt-Backend
weiterarbeiten, damit die Benutzeroberfläche, API, Tools und Agentenfunktion
nutzbar werden, **bis** Arctus reale Qualitätsgates besteht. Keine
stillschweigende Substitution bei fehlendem eigenen Checkpoint.

## Was ist heute wirklich vorhanden? — korrigiert nach aktuellem 350M-Stand

**Primärer Entwicklungsstrang: Arctus 350M, nicht 50M oder 100M.**
Das unabhängige Modell hat **349.825.152 Parameter** und verwendet einen
eigenen 16K-BPE-Tokenizer. Der Nutzer bestätigt eine bereits trainierte
Historie von **über zwei Milliarden verarbeiteten Tokens**. Frühere
Trainingsberichte nennen einen 2B-Fortsetzungsstand von ungefähr
**2,073 Milliarden Tokens** (späterer Bericht: 2.077.355.520).
Dies bezeichnet **historisch verarbeitete Tokens**, keine jetzt neu
trainierten Tokens und keinen bewiesenen Qualitätsstand.

**Wichtigster Unterschied:** der zuletzt erwähnte **1B-SAFE-Checkpoint**
ist nicht automatisch identisch mit dem jüngeren **2B-Foundation-
Fortsetzungscheckpoint**. Der 2B-Checkpoint darf nicht als SAFE,
chat-tauglich oder als release-geeignet bezeichnet werden, solange
Checkpoint-Digest, Lese-/Resume-Fähigkeit und Qualitäts-/Replayschutz
nicht im aktuellen Lauf kontrolliert sind.

**Quellcode ist vorhanden** – aber überwiegend auf dem separaten
Branch [`arctus350-chat-cloud-prep`](https://github.com/Cola968/neral-helios/tree/arctus350-chat-cloud-prep)
(auch `frontier/neral350-v1` und
`feature/arctus350-browser-knowledge-20261027`).
Beispiele:
- `frontier350/train350.py` – native 350M-Pretraining mit eigenem Resume
- `frontier350/train_chat350.py` – native Chat-/SFT-Versuche
- `frontier350/ARCTUS_2B_PLAN.md` – Foundation-Gates und 1B-Baseline
- `frontier350/CHAT350_RELEASE_GATES.md` – Posttraining-Gates
- `frontier350/POSTTRAINING_RESEARCH_2026_10_10.md` – dokumentierte
  Chat-SFT-Regression (NLL besser, Fähigkeiten 3/30 → 2/30)

Dass das Verzeichnis im `main`-Tree nicht vorhanden ist, war **kein
Beweis, dass der 350M-Trainingsstrang fehlt oder nicht existiert**.
Die vorangegangene Empfehlung `50M → 100M → 350M` als nötiger
Neustart war daher falsch. Keine Architekturänderung oder
Random-Reinitialisierung vor bestehender Checkpoint-Inventur.

**Historische 50M- und 100M-Dateien** bleiben Vergleichs- und
Architekturmaterial, sind aber **kein verpflichtender
Zwischenschritt für Arctus 350M**.

**Aktuelle Leistungsgrenze:** Zwei Milliarden Tokens machen Arctus
noch nicht zu einem verlässlichen Chatbot. Eigene
Pretraining- und frühe SFT-Berichte zeigen Schwächen bei
EOS/Antwortende, Wiederholung, exakter Instruktionsbefolgung
und freier Ausgabe. Ein 1.200-Schritt-SFT-Versuch verringerte
zwar den Validierungsverlust, verschlechterte jedoch die
30-Fragen-Fähigkeitsprobe von 3 auf 2; der Kandidat
darf deshalb nicht einfach befördert werden.
**NIS2 ist weiterhin nicht vollständig und gültig gemessen.**

## Welche Rolle darf Qwen noch spielen?

| Rolle | Zulässig? | Offenlegung |
|---|---|---|
| Vorübergehender NERAL-Chatbackend | Ja | „NERAL Assistant, derzeit mit Qwen3.5“ |
| Gegenmodell für getrennte Benchmarks | Ja | Werte separat ausweisen |
| Lehrer für **später gesondert freigegebene** synthetische Daten | Optional | Herkunft offenlegen, Qualität/Urheberrecht prüfen, keine kopierten Tests |
| Qwen-Checkpoint, -Embedding oder -LoRA als Arctus-Startgewichte | **Nein** | Verletzung der native-only-Linie |
| Qwen-Output als vermeintlich gemessene Arctus-Antwort | **Nein** | Technisch blockieren |
| Qwen-Fine-Tuning als „eigenes von Grund auf trainiertes Modell“ | **Nein** | Irreführende Modellherkunft |

Auch eigenes Training auf Qwen-generierten Daten könnte dessen
Antwortverhalten übernehmen. Wenn maximale Eigenständigkeit wichtig ist,
beginnen wir bewusst mit **human-/quellenbasierten, lizenzgeprüften Daten**,
einer von uns validierten Architektur und eigener Initialisierung;
synthetische Lehrer-Daten sind kein zwingender Bestandteil.

## GPU-First-Forschungsreihenfolge — 350M fortsetzen, nicht neu starten

**Stufe A: Native 350M-Bestandsaufnahme ohne GPU-Lauf.**
Branch `arctus350-chat-cloud-prep` als Basis der realen
350M-Quellen prüfen; 1B-SAFE und 2B-Fortsetzungscheckpoint in
einer versionsgebundenen Evidenzmatrix getrennt aufführen.
Bekannte Pfade/Hashes, Tokenstände, Tokenizer, Run-IDs, verwendete
Layer und Optimizerstatus aus GitHub-Artefakten erfassen, ohne
lokale Gewichte zu behaupten oder zu verändern. Fehlende Daten
als `not_verified` markieren.

**Stufe B: Kontrollierte echte RTX-4070-2B-Baseline.**
Erst bei eingeschaltetem PC Checkpoint/Tokenizer-Hashes,
Parameterzahl, CUDA, BF16, VRAM, tatsächliche trainierte Layers
und ein sehr kurzer reproduzierbarer Inferenz-/Eval-Test prüfen.
1B-SAFE und 2B-Foundation **mit gleichen Testbedingungen**
vergleichen. Kein automatisches Weitertrainieren; kein 1B-
Checkpoint durch 2B ersetzen ohne Freigabe und harte Qualitätsgates.

**Stufe C: Arctus 350M nachtrainieren und chatfähig machen.**
Nach Baseline auf einer separaten Fortsetzungs-/Posttraining-Linie
gezielt Antwortende/EOS, Anti-Repetition, deutsche Sprache,
Antwortqualität, Coding und dialogische Instruktionsbefolgung
verbessern. Train/Val/Holdout strikt trennen; Verschlechterung
des eingefrorenen Capability-Tests blockiert die Promotion.
Nur checkpointfähige kleine lokale RTX-4070-Versuche und
separat freigegebene GPU-Zeit. Keine fremden Qwen-Gewichte.

**Stufe D: Eigener NERAL-Chat & Coding-Agent.**
Eigene Arctus-Gewichte im Produkt **nur nach echten Chat-
Qualitätsgates** aktivieren. Parallel können wir UI und
sichere agentische Werkzeuge weiterentwickeln; sie verbessern
den Nutzwert, ersetzen aber nicht die Rohmodellleistung.
Qwen-Übergang muss bis dahin deutlich offengelegt werden.

## Strikte Release-Gates

1. **Provenienz:** random init / lückenlose eigene Checkpoint-
   SHA-256-Linie / reproduzierbare Tokendaten / keine Fremdgewichte.
   Manifest ist ein Audit-Hinweis, **kein kryptografischer Beweis**
   über die gesamte Trainingshistorie.
2. **Daten:** dokumentierte Quellen, Nutzungsrechte, PII-Filter,
   deduplizierte Train/Val/Test/Holdout-Splits, keine
   Evaluations-/Benchmark-Fragen im Training.
3. **Qualität:** echte, wiederholbare Rohmodellmessungen plus
   freie menschliche Dialogprüfung, keine Simulation als Modellscore.
4. **Zuverlässigkeit:** Antwort darf nicht durch unbelegtes
   „denkt nach“, externe Tools oder Qwen-Fallback besser erscheinen.
5. **Öffentliche Bezeichnung:** Nur solange die eigenen Gewichte
   wirklich aktiv sind und obenstehende Evidenz ausreicht:
   „NERAL/Arctus: eigenständig trainiertes Modell“. Eine frühe
   Demo darf separat „NERAL Assistant – Qwen-gestützte Beta“ heißen.

## Praktische Grenzen / Budget

Eigene GPU: **RTX 4070 mit 12 GB VRAM** (nicht immer eingeschaltet);
kurze checkpointfähige Piloten, keine schlaflose Dauerlast oder
erzwungenes Training mit überhitztem System. Hardwarediagnose separat.

**Maximal 25 € zusätzliche direkte Ausgaben insgesamt, Haushaltsstrom
zählt nicht.** Es gibt weiterhin **keine Freigabe**, Kosten auszulösen.
Ohne aktive RTX 4070 lässt sich hier Code/Datenaudit/Tests in GitHub
weiterführen, aber kein eigener 350M-Checkpoint auf dem
ausgeschalteten PC trainieren.

## Realismus

Ein eigenes 350M-Modell kann – selbst nach zwei Milliarden Tokens –
in allgemeiner Gesprächsqualität weit hinter Qwen3.5-9B liegen.
Kein Datum garantiert „ChatGPT-Qualität“. Der Meilenstein ist zuerst
eine **ehrlich eigene, funktionierende KI mit nachweisbar wachsender
Rohmodellkompetenz**; starke allgemeine Chatfähigkeiten benötigen
viel mehr Daten und Compute. Mehrwöchige Produkt-CI allein ist
kein Ersatz für Pretraining.
