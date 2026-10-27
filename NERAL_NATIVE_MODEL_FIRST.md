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

## Was ist heute wirklich vorhanden?

- **Eigene NERAL-Modelle (20M/40M/50M)** und nachgewiesener 50M-
  Sprachpilot mit 999.424 Pretraining-Tokens. Laut `MODEL_CARD.md`
  bestehen die freien Sprach-/Zuverlässigkeitsprüfungen noch nicht.
  Als fertigen Chatbot dürfen wir diese Modelle nicht bezeichnen.
- **100M Helix**: `model100.py` und `pretrain100_helix.py` im
  Hauptbranch, inklusive zufälliger Gewichtsinitialisierung,
  Gradient-Checkpointing, BF16/CUDA-Gate, Training/Validierung und
  Checkpoint-Fortsetzung. Das Vorhandensein des Codes **beweist keinen
  erfolgreich trainierten 100M-Checkpoint**.
- **Arctus350**: GPU-Workflows für Datenvorbereitung und Chatpilot
  vorhanden, aber `frontier350/train_chat350.py` und weitere
  `frontier350`-Quellen im heutigen **main** nicht enthalten.
  Mögliche separate Branches/lokale Quellen müssen explizit
  nachvollzogen werden; 350M daher nicht als reproduzierbar startbereit
  oder als bereits erfolgreiches Chatmodell melden.
- **NERAL-P1 SFT-Drafts (PR #139)** gehören zur möglichen
  *Qwen-abgeleiteten Produktlinie*, nicht automatisch zu Arctus.
  Für Arctus brauchen wir einen bewusst getrennten Datenaudit und eine
  passende, unabhängig initialisierte Modell-/Tokenisierungs-Pipeline.
- **NIS2 ist für Arctus ohne vollständige Originalmodell-Evidence nicht
  ermittelt.** Weder Qwen-Scores noch Tool-Antworten fließen ein.

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

## GPU-First-Forschungsreihenfolge

**Stufe A: 20M/50M zuverlässig messen.** Kein Training nach Gefühl.
Preflight von Treiber, VRAM, Speicher, Tokenizer und Checkpoint-Digest,
kurze freie Generierung, Wiederholungsrate, EOS, val loss, Sprachqualität,
korrigierbare Fehler und eindeutig getrennte Tool-Ergebnisse.

**Stufe B: 100M Helix reproduzierbar betreiben.** CPU-only
Konfigurations-/Smoke-Tests auf GitHub; auf RTX 4070 einen explizit
freigegebenen kurzen BF16-Pilot mit `pretrain100_helix.py` ausführen.
Verlustkurve, echte freie Antworten, NIS2-Evidence-Status, VRAM und
Checkpoint-Resume protokollieren. Nur bei besserer Qualität sukzessive
mehr *lizenzierte* Tokens investieren. Ein guter Loss allein reicht nicht.

**Stufe C: Arctus350 reproduzierbar machen.** Existierende
`frontier350`-Quellen aus identifizierten Branches/lokalen
Arbeitsständen sichern und Tests/CIs zum Hauptprojekt ergänzen,
Datensplits/Lizenzen/Tokenizer-Herkunft prüfen; erst dann
350M-Trainingsversuch freigeben. Ein größeres Modell ohne ausreichend
Daten und Compute wird nicht automatisch besser.

**Stufe D: Chatfähigkeit und Agentenwerkzeuge.** Wenn das eigene
Grundmodell schon echte, unabhängige Rohmodell-Baselines besteht:
eigene geeignete Konversationsdaten und ggf. SFT/Preference-Training
auf **Arctus-Gewichten**. Calculator, Projektgedächtnis, Dateizugriff
und Coding-Agent sind separat autorisierte Softwarewerkzeuge,
keine vermeintlichen Fähigkeiten im Grundmodell. Kein Eigenmodell
wird automatisch zum produktiven Backend befördert.

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
weiterführen, aber kein eigener 100M/350M-Checkpoint auf dem
ausgeschalteten PC trainieren.

## Realismus

Ein eigenes 100M/350M-Modell kann – selbst nach viel lokaler Arbeit –
in allgemeiner Gesprächsqualität weit hinter Qwen3.5-9B liegen.
Kein Datum garantiert „ChatGPT-Qualität“. Der Meilenstein ist zuerst
eine **ehrlich eigene, funktionierende KI mit nachweisbar wachsender
Rohmodellkompetenz**; starke allgemeine Chatfähigkeiten benötigen
viel mehr Daten und Compute. Mehrwöchige Produkt-CI allein ist
kein Ersatz für Pretraining.
