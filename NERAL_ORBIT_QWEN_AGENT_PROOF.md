# NERAL Orbit × Qwen: echter, begrenzter Agententest, 1. November 2026

**Beleg:** [Windows-Run 41308495223](https://github.com/Cola968/neral-helios/actions/runs/41308495223),
erfolgreich. **27/27 Python-Tests** für Assistent,
Planung, API-Schutz und Pipeline bestanden.

Der Auftrag war eine öffentliche Recherche zum Flughafen
Berlin Brandenburg. Über den PC-Runner wurde **das tatsächlich
installierte Qwen3.5-9B über lokales Ollama** benutzt, nicht
das native Arctus350-Modell. Zwei separate Freigaben waren
im einmaligen Workflow ausdrücklich gesetzt.

1. `neral_agent_plan.propose_agent_plan` verwendete den
   bereits vorhandenen und lokal auf Loopback beschränkten
   `neral_assistant.assistant_turn` **ohne angebotene Tools**.
   Qwen lieferte einen validierten Plan mit **drei Schritten**.
2. Der neue `neral_orbit_agent_loop` akzeptiert nur einen
   validierten, noch nicht ausgeführten Plan. Keine frei
   formulierten Toolnamen werden als Aktionen interpretiert.
3. Das ursprüngliche `neral_orbit_research.run_public_research`
   aus **Draft #176** führte genau **einen öffentlichen
   Wikipedia-Request** aus. Er lieferte **drei Quellen**.
4. Die Originalimplementierung `OrbitStore` aus Draft #176
   verwaltete die Aufgabe im temporären SQLite:
   `DRAFT → APPROVED_NO_EXECUTOR → RESEARCH_RUNNING → REVIEW_READY`.
5. Ein zweiter lokaler Qwen-Aufruf erzeugte aus den gefundenen
   öffentlichen Auszügen einen **unveröffentlichten Entwurf**.
   Eine unabhängige Faktenprüfung wurde **nicht** behauptet.

Der abschließende maschinenlesbare Status meldete
`plan_validated=true`, `plan_step_count=3`,
`research_completed=true`, `source_count=3`,
`draft_generated=true`, `requires_human_review=true`,
`auto_published=false` und `model_weights_modified=false`.

**Privatsphäre:** Weder generierte Texte noch Quellenpassagen,
Modellgewichte oder SQLite-Datenbanken wurden als GitHub-Artefakte
hochgeladen. Das temporäre SQLite-Testverzeichnis wurde am Ende
entfernt; daher steht der Probebericht nicht dauerhaft in
deiner regulären Helios-Installation.

## Was implementiert wurde

- `neral_agent_plan.py`: Strikte JSON-Pläne von Qwen3.5,
  maximal vier erlaubte Schrittarten, keine Shell und
  keine automatische Freigabe.
- `POST /api/assistant/agent-plan`: Session-/Origin-
  geschützter lokaler, separat bestätigter Plan-Entwurf.
  Ohne `NERAL_ASSISTANT_ENABLED=1` deaktiviert.
- `neral_orbit_agent_loop.py` und
  `neral_orbit_agent_smoke.py`: einmalige Komposition
  zwischen beiden bestehenden Draft-Branches.
- Die GitHub-Aktion checkt Draft #203 und Draft #176 in
  **getrennte Verzeichnisse** aus. Sie merged nichts und
  lädt weder Modelle noch Daten herunter.

## Noch ausdrücklich offen

Dies ist **eine getestete einmalige Integration**, noch
**keine aktivierte autonome Agentenfunktion in der
Orbit-Weboberfläche**. Die beiden PRs bleiben separat,
und die Ablauf-Steuerung muss nach fachlichem Review in
einer gemeinsamen Codebasis integriert werden.
Ein dauerhaft laufender Agent, Scheduler, autonomes Browsen,
allgemeine Toolwahl, Live-News-Recherche und Arbeiten ohne
Bestätigung sind nicht implementiert. Dieser Test belegt
auch nicht die sachliche Richtigkeit des generierten Entwurfs.
