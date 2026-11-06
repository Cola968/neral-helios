# NERAL Agentenmodus V1 — 50 technische Verbesserungen

Stand: 6. November 2026. In der Helios-Oberfläche ist jetzt ein eigener
**Agent**-Bereich vorgesehen. Die Agentenwerkzeuge laufen lokal und ohne
Cloud-Inferenz. Der Ablauf ist regelgesteuert, **nicht** von einem autonom
programmierenden Sprachmodell gesteuert. Die Modellgewichte bleiben unverändert.

## Benutzung

Agent öffnen → Ziel und eine der sechs Strategien auswählen →
„Plan ausführen“ oder „Nächster Schritt“. Ergebnisse werden mit Status,
Werkzeugname und Fortschritt angezeigt. Für Änderungen im aufklappbaren
Bereich den relativen Dateipfad und den **vollständigen neuen Dateiinhalt**
angeben, den Diff prüfen und ausdrücklich freigeben. Einmalige Rücknahme
ist bei unverändertem Dateistand möglich.

## Implementierte Eigenschaften

Diese 50 Punkte umfassen neue Funktionen, Grenzprüfungen und Bugfixes.
Sie sind keine 50 unabhängigen KI-Modelle oder Plug-in-Anbindungen.

1. Agentenansicht als eigener UI-Bereich.
2. Vorherige Ansicht bei Rückkehr wiederherstellen.
3. Sechs feste Strategien: audit, quality, search, inspect, docs, security.
4. Freitextziel mit Längenbegrenzung.
5. Optionaler Suchtext.
6. Optionaler relativer Dateipfad.
7. Fest definierter Werkzeugplan.
8. Einzelne Schritte ausführen.
9. Gesamten Plan mit begrenzter Schrittzahl ausführen.
10. Fortschritt je Schritt anzeigen.
11. Strukturiertes Resultat je Werkzeug.
12. Ereignisprotokoll der Schritte.
13. Laufende Aufträge pausieren.
14. Pausierte Aufträge fortsetzen.
15. Aufträge sicher abbrechen.
16. Zeitbudget für Aufträge.
17. Feste Tool-Allowlist statt dynamischer Befehle.
18. Obergrenze gleichzeitiger Aufträge.
19. Freigewordene Auftragsslots wiederverwenden.
20. Ungültige Auftrags-IDs zurückweisen.
21. Arbeitsverzeichnis strikt begrenzen.
22. Traversal mit ../ blockieren.
23. Absolute Pfade blockieren.
24. Windows-Laufwerkspfade blockieren.
25. Windows-Gerätenamen blockieren.
26. Backslashes und Prozentkodierung in Pfaden blockieren.
27. Symlinks blockieren.
28. Versteckte Dateien schützen.
29. Secret-/Token-Dateinamen ausfiltern.
30. Modell-, Cache- und Dependency-Ordner beim Scan überspringen.
31. Höchstzahl gescannter Dateien.
32. Maximale lesbare Dateigröße.
33. UTF-8-Textvalidierung.
34. Dateiinventar und Erweiterungsstatistik.
35. Python-Syntaxprüfung ohne Ausführung.
36. Python-Funktions-, Klassen- und Importzählung.
37. JSON-Syntaxprüfung.
38. TOML-Syntaxprüfung.
39. CSV-Header-Prüfung.
40. TODO-/FIXME-/XXX-Treffer mit Zeile.
41. Abschließenden Leerraum melden.
42. Überlange Zeilen melden.
43. Fehlenden abschließenden Zeilenumbruch melden.
44. Lokale Markdown-Links auf Existenz prüfen.
45. Verdächtige dynamische Codeausführung markieren.
46. Literale Textsuche ohne Regex-Ausführung.
47. Treffer und Ausgabemengen begrenzen.
48. Datei-SHA-256 und begrenzte Textvorschau.
49. Diff-Vorschau, Freigabetoken und atomare Dateiänderung mit SHA-Konfliktschutz.
50. Undo nur bei unverändertem Stand seit dem Schreibvorgang.

Weitere Korrekturen: Unveränderte Dateiinhalte werden nicht erneut
geschrieben. Die Freigabe ist nach Zeitablauf unmöglich. Bei Abbruch
wird ein vorgemerkter Dateipayload verworfen. Ein gescheiterter
Schreibvorgang bewahrt die Freigabe für einen sicheren Wiederholungsversuch.
Bearbeitbare Endungen und Inhaltsgrößen sind begrenzt. Alle Agenten-
API-Routen unterliegen dem bestehenden lokalen Session-, Host-, Origin-
und Rate-Limit-Schutz der Helios-App.

## Einschränkungen

Das ist **kein autonomer Coding-Agent mit lokalem Qwen/NERAL-Modell**
und kein 24-Stunden-Worker. Keine Shell, keine Installation fremder
Abhängigkeiten, kein Netzabruf, keine automatische Ausführung von Tests
oder Änderungen an GitHub. Abbruch wirkt zwischen Werkzeugschritten.
Aufträge und Undo-Payloads sind nur im Arbeitsspeicher des Servers;
nach Neustart sind sie nicht verfügbar. Änderungen an Dateien bleiben.

Ein SHA-256-Vergleich dient nur dem Versionsschutz, nicht als
Authentifizierung. Das Projekt benötigt weiterhin eine separat
evaluierte lokale LLM-Tool-Call-Schicht, einen persistenten Auftrags-
speicher und vollständige Ende-zu-Ende- und Windows-Tests.

## Regressionstests

```bash
python -m pip install pytest Flask
python -m pytest -q tests/test_agent_mode.py tests/test_agent_api.py
node --check static/agent-mode.js
node --test tests/test_agent_ui.cjs
```

Im PR läuft `.github/workflows/neral-agent-mode-ci.yml`.
Die bestehende Helios-Testpipeline ist zusätzlich relevant.
