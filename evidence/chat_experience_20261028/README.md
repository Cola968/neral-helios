# Chat Experience & Stability: ausgeführte Prüfungen

Basis: `main` / `4e4dca3da0d8bc0d3ee5ec697edb7575cf9aa278`.
Eigener Branch: `codex/neral-chat-experience-stability`.
Details und Startbefehle: [Update-Dokumentation](../../docs/CHAT_EXPERIENCE_UPDATE.md).

## Nachgewiesene Ergebnisse

- **113 relevante Python-Tests bestanden:** echter Flask-Handler, Session-/Origin-
  Schutz, Rechenrouting, Provenienz, ChatStore, Projekte, vorhandene Streaming-/
  Gateway-Integration. Native Antworten in API-Tests verwenden ausdrücklich einen
  injizierten Token-Produzenten, keine echten Modellgewichte.
- **16 Node-Tests bestanden:** Kontextgrenzen und Unvollständigkeit, Herkunft,
  Import/XSS/Markdown, Stream-Abschluss mit/ohne Zeilenumbruch, fehlerhafte oder
  doppelte Abschlüsse, widersprüchliche Quellen und verständliche Formatfehler.
  Zwei zusätzliche RED-/GREEN-Fälle sichern die Migration alter Import-Chats
  und die Ablehnung von Prototype-Eigenschaftsnamen als vermeintliche Modellnamen.
- **Drei Browserläufe bestanden:**
  - Bestehende UI-Regressionen: Theme, Entwurf/Reload, Import, Markdown/XSS,
    Varianten, Löschen, Modellmenü, Navigation, Projekte, Einstellungszustände.
  - Bestehender Workspace-Test, erweitert: gleichzeitige Gerätefassungen,
    Konfliktkopien, lokale Entwürfe, Streaming, Teilantwortfehler, Stoppen,
    Erhalt laufender Antworten bei neuem Rendern und paralleler nächster Entwurf.
    Dieser Lauf verwendet eine explizite lokale Test-API für Modelltexte.
  - **Neuer echter App-Einstieg:** startet `app.py` in einem isolierten Ordner
    ohne Checkpoints, CPU, Vorbereitungsmodus, echter Cookie-/Origin-Schutz.
    Keine Rechen-API-Mocks. Rechnet 7+8=15, DE/EN 57+68=125 und `(7+8)*2=30`;
    unsupported Compound bleibt ohne Teilergebnis. Prüft Fehlermeldung,
    Fragen-/Entwurfserhalt nach Reload, externe Importherkunft, XSS, Tastatur,
    Byte-Limit auch beim Moduswechsel, Leseposition und Sprung zur neuesten
    Nachricht. Breiten 360/390/768/1440, zusätzliche kurze Mobilhöhe 480 Pixel.
- Kompilierung und `git diff --check` bestanden.

## RED → GREEN

Der erste neue API-Lauf hatte 14 Fehler und 11 bestandene Fälle; die bisherigen
Chat-Antworten hatten keine verlässliche Quellenkennung und natürliche DE/EN-
Rechenanfragen erreichten das nicht geladene Modell. Der echte Browser-RED-Test
scheiterte am deaktivierten Senden im Werkzeugmodus. Die neuen History-/Provenienz-
Tests reproduzierten die Aufnahme abgebrochener/importierter Antworten.

Weitere RED-Läufe belegten zusätzliche übergangene Aufgaben nach Zeilenumbrüchen,
verschwindende laufende Nachrichten beim Rendern, fehlende `solve.result`-
Kompatibilität sowie unverständliche Parserfehler. Diese Ursachen wurden behoben
und die jeweiligen Prüfungen anschließend wiederholt. `validation.txt` enthält
die zusammengefassten tatsächlichen Runner-Ausgaben, keine erfundenen Messwerte.

Bei der abschließenden Startkettenprüfung wurde außerdem der alte Cookie-Fehler
in `launcher.py` reproduziert: Der echte laufende Server lieferte ohne den über
`/` ausgestellten Sitzungscookie HTTP 401 für `/api/info`; `launcher.status()`
gab deshalb fälschlich False zurück. Nach der Korrektur verwendet die Prüfung
einen privaten Cookie-Opener für beide Requests. Der vollständige echte Browser-
Smoketest prüft nun auch diesen Statusaufruf in einem separaten Python-Prozess
gegen den laufenden isolierten Flask-Server. RED und anschließender GREEN wurden
ausgeführt. Keine Token-Datei wird dafür gelesen oder Authentifizierung abgeschaltet.

Der erste abschließende Linux-CI-Lauf
[`40760274731`](https://github.com/Cola968/neral-helios/actions/runs/40760274731)
bestand Python-, Node-, bestehende Browser- und Startertests, fand aber einen
echten Layoutfehler in der neuen 390×480-Mobilprüfung: Andere Schriftmetriken
konnten den Senden-Button unter den sichtbaren Rand schieben. Der kurze mobile
Viewport blendet jetzt den Einstiegsblock aus und begrenzt das Eingabefeld.
Der Test bleibt unverändert streng; es wird keine größere Bildschirmhöhe als
Umgehung gewählt. Die Screenshots bei normaler Höhe bleiben repräsentativ.

Alte importierte Gespräche ohne Herkunftsflag pro Nachricht werden beim Laden
und Synchronisieren konservativ migriert. Neue echte Antworten werden explizit
als nicht importiert gespeichert und bleiben davon unterscheidbar. Der bestehende
Browserlauf lädt zusätzlich einen solchen alten Gesprächsstand neu und prüft
Label und ausgeschlossenen nativen Kontext.

## Breiter Repository-Lauf und offene Grenzen

`python -m pytest -q` wurde ebenfalls ausgeführt. Zuerst: 423 bestanden,
33 fehlgeschlagen, einer übersprungen. Die alten API-Testfixtures verwendeten
noch keine lokale Sitzung, ein überholtes gemeinsames Rate-Limit und einen
unvollständigen History-Fall für eine Größenprüfung. Die Fixtures sind jetzt
authentifiziert und prüfen den aktuellen Vertrag; die Produktionssicherheit
wurde nicht gelockert. Das bisherige strukturierte `solve.result` bleibt erhalten.

Der abschließende vollständige Lauf: **451 bestanden, 5 fehlgeschlagen, 1 übersprungen**.
Alle verbleibenden Fehler benötigen ausdrücklich ausgeschlossene lokale Artefakte:

- `tests/test_curriculum_v2.py::test_v2_splits_and_hashes`: `curriculum_v2/manifest.json`.
- `tests/test_curriculum_v2.py::test_pilot02_lineage`: `export/pilot_02/neral20m.pt`.
- `tests/test_growth.py::test_40m_exact_count_and_inherited_weights_unchanged`:
  `export/neral40m/neral40m.pt`.
- `tests/test_regressions.py::test_curriculum_disjoint_and_masked`:
  `curriculum/train.jsonl` (lokale Korpora).
- `tests/test_regressions.py::test_original_untouched_and_parameter_count`:
  `seed/neral20m.pt`.

Diese Fehler werden weder als grün ausgegeben noch durch gefälschte Artefakte
oder stilles Herunterladen umgangen. Der vorhandene Korpus-Test mit expliziter
`skip`-Bedingung bleibt übersprungen. Die CPU-App-CI führt die relevanten und
artefaktfreien zusätzlichen API-Regressionen aus.

Keine echte Modellantwortqualität, nativer Modell-Durchsatz oder 350M-Integration
wurde gemessen oder behauptet. Der Stoppen-Button ist mit dem tatsächlichen
Transport und einem kontrollierten Tokenproduzenten geprüft, nicht mit einem
privaten nativen Modell. Mobil bedeutet hier Browser-Viewport, kein physisches
Android-/iOS-Gerät mit realer Bildschirmtastatur.

## Screenshots

`desktop.png`, `mobile.png`, `error.png` stammen vom **echten Flask-Einstieg**
mit ausschließlich synthetischen Rechen-/Fehlerfällen und leeren Modellpfaden.
Sie enthalten keine privaten Chats, Gewichte, Tokens oder Zugangsdaten.
Keine GPU-Trainingsläufe, HF-Compute-Jobs, Deployments, Merges oder geplanten
PC-Jobs wurden gestartet. Der Design-Generator wurde wegen seiner Kosten nicht verwendet.
