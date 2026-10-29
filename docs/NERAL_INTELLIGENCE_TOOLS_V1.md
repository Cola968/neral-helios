# NERAL Intelligence – überprüfbare Werkzeugfähigkeit v1

## Vorhandenes Werkzeug auf die gemeinsame Chatbasis übernehmen

Dieser Zweig übernimmt die vorhandene Implementierung aus Draft #177 auf die
Qwen-/Arctus-Chatbasis #197. Im Modus „Eigene Modelle“ kann der Flask-Chat
damit ausgewählte mehrstufige Rechenaufträge beantworten: „Berechne 2+2 und
ziehe 1 ab“ → **3**. Weitere vollständig unterstützte Formen: EN Prozent plus
festen Summanden; DE/EN vollständig formulierter Multiplikationsfolgeschritt.
Der genaue Zahlenwert wird mit `fractions.Fraction` und dem bestehenden eng
begrenzten AST-Rechner berechnet, nicht mit `eval`, `exec` oder
Sprachmodell-Vermutungen. Der Qwen-Assistant-Modus bleibt unverändert und
verwendet weiterhin seine eigene Werkzeug-Schnittstelle.

**Wichtig:** Das verbessert das **Produktverhalten**, nicht die gelernt-intelligenten Gewichte von Arctus350 oder der Byte-Modelle. Jede Erfolgsmeldung weist `model_generated:false` und `source:tool:calculator` aus. Es entsteht keine neue Modelldatei, kein SFT-Checkpoint und kein NIS2-Ergebnis.

## Sicherheitsgrenzen

- Der Parser akzeptiert nur eine eng umschriebene DE-/EN-Grammatik und prüft die **gesamte** Eingabe; freie Folgetexte oder zweite Anweisungen werden abgelehnt, nicht aus einem Präfix beantwortet.
- Aufrufe unterliegen den gleichen lokalen Cookie-, Host-, Origin-, Inhalt- und Rate-Grenzen wie der Chat.
- Keine freie Python-Ausführung, kein Shell, keine externen Provider, kein Scraping, keine Kosten.
- Umfangreichere Sachfragen laufen weiterhin über den vorhandenen Modellpfad oder erzeugen – ohne geladenes Modell – eine ehrliche Fehlermeldung.

## Prüfungen

Das Original #177 bestand bereits Flask- und Browser-CI auf seiner damaligen
Basis. Für die hier neu kombinierte Basis wurden fünf lokale Standardbibliotheks-
Tests inklusive 300 unabhängiger Zahlenbeispiele und Routing mit dem echten
`generate`-Funktionskörper ausgeführt. Die lokale Routing-Prüfung ersetzt
Flask, Cookies, Origin-Schutz und Netzwerkanbindung durch einen Testadapter;
sie ist **kein** HTTP- oder Browser-End-to-End-Test. Diese Tests sind für eine
spätere CPU-CI in `neral-qwen-unified.yml` ergänzt; auf diesem Draft wurde CI
bewusst nicht gestartet.

## Was als Nächstes für das eigene Modell wichtig bleibt

Das experimentelle native Arctus350 hatte auf dem neuen 50–99-VAL-Audit **0/32** korrekt. Das ist nicht durch bessere Parser weggehebelt. Sinnvolle nächste native Schritte nach Rückkehr des PCs: sorgfältig getrennte und lizenzgeprüfte DE/EN-Instruktionsdaten, unveränderte echte TEST/NIS2-Gates, reproduzierbare Baseline und nur kontrollierte kurze GPU-Piloten mit klaren Abbruchkriterien. Die vorbereiteten v3-Column-Carry-Daten sind bereits verfügbar und sollten nicht durch ungezieltes Wiederholen desselben SFT-Laufs entwertet werden.

**Release:** Draft auf #197 → #196 → #175 → #173; kein Merge/Deployment oder GPU-Job.
