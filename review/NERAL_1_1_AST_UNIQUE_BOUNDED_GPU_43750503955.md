# NERAL 1.1 – AST-Reparatur, unabhängige Logik und begrenzter Qwen-Fallback

**Datum:** 15. November 2026. **Status:** Tatsächlich ausgeführter RTX-4070-Lauf, GitHub Actions vollständig grün. Dieser Test ist *kein* Blindtest.

## Versions- und Beweiskette

- Getestete NERAL-Quelldateien, unveränderlich ausgecheckt: `fbc6d30d7645f2621d1cda17b0226fba2b162e98`.
- Feature-Branch: `feat/neral-1-1-ast-logic-bounded-fallback-20261115`.
- `main` weiterhin `7fb05452fa7588191f9a127b11511a2ec312baf1`.
- CPU-Gate: https://github.com/Cola968/neral-helios/actions/runs/43750343200 – **246/246 bestanden**, einschließlich neuer, vorher nicht verwendeter synthetischer Fälle.
- GPU-Gate: https://github.com/Cola968/neral-helios/actions/runs/43750503955 – **Erfolg**, einschließlich CPU-Preflight, echter RTX4070/Docker-Ausführung, Artefakt-Upload, Roundtrip-Download und Bytevergleich.
- Vollständiges ZIP-Rohartefakt: https://github.com/Cola968/neral-helios/actions/runs/43750503955/artifacts/14176685713
- ZIP-SHA256: `78d85834c2232a0efd2be72711abf5f62ab660eaa3c015d1702e9bd8741dd33a`.
- Aufgabendefinitionen (zehn Fälle) SHA256: `d4ce4bdc88884f5455d446fc41e995696e72035e9c5a182d5ea13c46cb0dd872`.
- Aufgabendefinitionen **der ursprünglichen acht** sind mit SHA256 `432148ba99fe507a60cdd619a68b6b69a985789544c9c63ed29192d8e04d5de8` identisch und wurden im CPU-Preflight bestätigt. Zwei zusätzliche GPU-Fälle sind neu und werden separat ausgewertet.
- **Kein** Training, **keine** Änderung an main, **kein** Öffnen/Verändern/Neugenerieren eines Blind100-Datensatzes.

## Reale Messwerte je Fall

Alle Zeitangaben sind echte HTTP-End-to-End-Wandzeiten im GPU-Runner (Modellladezeiten, Kandidaten, Docker und Fallback enthalten). Die Z3-Fälle nutzen lokale CPU/Z3-Beweise und brauchen keinen QLoRA-Aufruf.

| Fall | Route | Reparaturen | Vollständig verifiziert | Endantwort | Wandzeit |
|---|---|---:|---|---|---:|
| s01 Graphfärbung, bisheriger Diagnosefall | direkt Z3 | 0 | JA | 30 | 0,087 s |
| s02 Positionscode, bisheriger Diagnosefall | direkt Z3 | 0 | JA | 12 | 0,248 s |
| s03 Allgemeine Wahrheit/Lüge zählen, bisheriger Diagnosefall | begrenzter Qwen-Fallback | 0 | NEIN | explizite Ablehnung | 13,144 s |
| s04 Allgemeine Zuordnung zählen, bisheriger Diagnosefall | begrenzter Qwen-Fallback | 0 | NEIN | explizite Ablehnung | 12,797 s |
| s05 Mehrstufige Mathematik I, bisheriger Diagnosefall | Python + unabhängiger Rechenprüfer | 0 | JA | 596 | 9,843 s |
| s06 Mehrstufige Mathematik II, bisheriger Diagnosefall | Python + gezieltes print-Repair + Prüfer | 1 | JA | 163 | 6,350 s |
| s07 Kombinatorik, bisheriger Diagnosefall | Python + gezieltes assert-Repair + Prüfer | 1 | JA | 0 | 5,108 s |
| s08 Mehrdeutige Zusatzregel, bisheriger Diagnosefall | begrenzter Qwen-Fallback | 0 | NEIN | explizite Ablehnung | 22,420 s |
| s09 **Neues** Wahrheitsrätsel (4 Personen) | Python, zwei Versuche; dann Fallback | 1 | NEIN | explizite Ablehnung | 57,288 s |
| s10 **Neue** eindeutige Zimmerzuordnung (5 Personen) | Python + unabhängiger Eindeutigkeitsprüfer | 0 | JA | 2 | 19,590 s |

**Kennzahlen, ursprüngliche acht:** 5/8 unabhängig verifiziert; Median **8,097 s**, Mittel **8,750 s**. Davor beim Z3-First-Lauf: 3/8 und Median 21,248 s. Beide zuvor festgelegten Zielwerte (mindestens 5/8, Median <15 s) sind auf dieser **bereits mehrfach betrachteten** Diagnosematrix jetzt erreicht. Der Wert ist **keine** externe Erfolgsprognose.

**Neue Fälle:** 1/2 unabhängig verifiziert, Median **38,439 s**. **Alle zehn:** 6/10 unabhängig verifiziert, Median **11,320 s**, Mittel **14,687 s**. Es wurden 8 reale QLoRA-Kandidatenaufrufe, 5 Docker-Ausführungen und 4 begrenzte Qwen-Fallbacks registriert.

## Nachgewiesene Ursachen und Grenzen

1. **Leeres Schema-Feld:** Der vorherige QLoRA-Structured-Output-Proposer verwarf ein Python-Paket mit zusätzlichem `"schema": {} schon beim Parser. Jetzt wird *ausschließlich* ein exakt leeres, korrekttypisiertes Schema-Feld bei Python-Vorschlägen ignoriert. Nichtleere und unbekannte Felder werden weiterhin abgewiesen. Bei s06 wurden vor der Reparatur `strategy,code,schema` und danach ebenfalls solche Felder ausgegeben; beide wurden korrekt normalisiert.
2. **AST-Reparatur erreicht:** s06 begann mit assert, aber ohne abschließendes print; der einzige Reparaturaufruf ergänzte die Ausgabe. s07 begann mit print, aber ohne assert; der einzige Reparaturaufruf ergänzte die Gegenprüfung. Beides benötigte **jeweils nur eine Docker-Ausführung** und wurde erst nach unabhängigem Volltextbeweis freigegeben.
3. **Eindeutigkeit nur bei unabhängigem Beweis:** s10 sammelte gültige Belegungen in einer ausführbar aufgerufenen Python-Funktion, führte dort `assert len(valid_assignments)==1` aus und gab Milos Zimmernummer `2` aus. Der *separate* vertrauenswürdige Parser enumerierte alle Zuordnungen aus dem Originaltext und bestätigte dieselbe einzigartige Belegung.
4. **Neue Wahrheit/Lüge-Aufgabe scheitert:** s09 erzeugte einen falschen Logiktest, der in Docker mit AssertionError abbrach. Ein Reparaturversuch und der begrenzte Qwen-Fallback lieferten keinen unabhängig zulässigen Abschluss. NERAL gab **keine** unbewiesene Zahl als verifiziert frei. Dieser Fehler demonstriert die fortbestehende Schwäche bei freier Logikübersetzung.
5. **Fallback-Kürzung ist keine Latenzgarantie:** Die separaten Qwen-Aufrufe benutzen im experimentellen Pfad 640 Tokens und `think:false`. s03, s04, s08 und s09 endeten jeweils mit `done_reason=length` und expliziter Ablehnung. Einzelne End-to-End-Zeiten blieben 22,42 bzw. 57,29 Sekunden. Ein kürzeres Budget kann die subjektive Antwortqualität verschlechtern, auch wenn es den Median senkt.
6. **Keine universelle Selbstprüfung:** Modellgeschriebene `assert`-Zeilen sind nur Ausführungskontrollen. Den Status `safe_to_promote=true` erteilt weiterhin ausschließlich ein vertrauenswürdiger Parser mit vollständiger unabhängiger Enumeration/Z3-Äquivalenz oder exakter Arithmetikprüfung. Alle unbekannten Fälle bleiben unverifiziert.
7. **Nicht blind:** Die ursprünglichen acht Aufgaben sind dem Entwicklungsprozess bekannt und wurden mehrfach zur Fehleranalyse verwendet. Die zwei weiteren Fälle waren neu für die GPU-Abnahme, wurden aber ebenfalls nur einmal in einem kleinen Diagnosesatz geprüft. Keine nach außen generalisierbare Null-Regression- oder Qualitätsgarantie.

## Nächste fachlich sinnvolle Arbeiten (nicht durchgeführt)

- Flexible Aussage- und Zuordnungslogik über neue, disjunkte Satzgrammatiken unabhängig verifizieren; niemals freie Modelllogik über modelleigene Assertions zertifizieren.
- Die vier 640-Token-Fallback-Abbrüche gegen vollständig beantwortete unbeschränkte Qwen-Texte mit **neuen** Testfällen als Qualitäts-/Latenz-Pareto vergleichen. Kein blindes weiteres Kürzen.
- Eigene, noch nicht sichtbare Tests pro Subdomäne und neu versiegelte Blindprüfung erst nach getrenntem Review. Keine stillschweigende Übernahme nach `main`.
