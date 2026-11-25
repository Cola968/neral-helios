# V4.3 Websearch: isolierter CPU-Fix

Ausgangsbasis: `main` `5877b73851c986f39f4dbeb470cd8fb08bd2707d`.
Änderungszweig: `fix/v43-websearch`. Kein Merge, kein GPU-Lauf,
keine Änderungen an Trainingsgewichten, Blind200, Reference50 oder ObsidianHard.

## Ursachen im Stand vor diesem Fix

- `neral_local_wiki.source_required()` erkannte im Chat vor allem
  ausdrückliche Quellenwörter. Aktuelle Nachrichten, historische Faktenfragen
  oder einfache ausdrückliche Suchen lösten nicht zuverlässig die Webroute aus.
- Im V4.3-Draft-first-Profil generierte
  `ThreeTierRuntime.answer_v43_bounded()` die Modellantwort **vor** dem
  optionalen `neral_v43_routes._web()`-Abruf. Der Web-Reader lieferte
  nur Metadaten an die nachträgliche Prüfung; der generierte Text wurde nicht
  auf Grundlage der Webquellen erzeugt. Ausnahmen wurden dort verworfen.
- Die Vorschau-Oberfläche erwartete für jede 9B-Antwort das
  `qwen3.5:9b+neral-v2-full`-Adapterprofil. Ein explizites
  `--v43-profile` mit rohem `qwen3.5:9b` wurde deshalb abgelehnt.
- Der reine Browser-Suchpfad zeigte teilweise ausschließlich
  Suchmaschinen-Snippets. HTML-Parser-Änderungen und Providerfehler konnten
  zu leeren Ergebnissen führen, ohne einen aussagekräftigen End-to-End-Status.

## Verhalten nach dem Fix

- Im interaktiven Drei-Stufen-Chat wird eine öffentlich formulierte
  explizite Suche, aktuelle Zeit-/Nachrichtenfrage oder gezielte
  Faktenfrage automatisch gesucht. Private Eingaben, Berechnungen,
  Programmieraufträge, Anhänge mit Sonderverarbeitung und Benchmark-Aufträge
  werden nicht automatisch übertragen.
- `benchmark_isolated=true` verhindert providerseitige Suchaufrufe auch bei
  einem mitgesendeten `web_query`. Isolierung ist an die HTTP-Anfrage
  gebunden und wird per `ContextVar` wieder freigegeben.
- Im V4.3-/Amber-Pfad werden nur **wirklich gelesene** HTTPS-Seiten vor
  der Modellausgabe in einen begrenzten, ausdrücklich als unvertrauenswürdig
  gekennzeichneten Nutzerkontext eingefügt. Metadaten-/Snippet-Treffer
  reichen nicht als inhaltlicher Beleg.
- Wenn Quellen nicht abrufbar sind, kommt ein sichtbarer `503`-Fehler
  mit klassifiziertem Status statt einer stillen Antwort aus Modellwissen.
- Die Suche verwendet die bestehende feste DDG-HTML/Bing-RSS/Google-News/
  Wikipedia-Kette. DDG-HTTP-202 führt zu einem einzigen begrenzten
  Wiederholungsversuch; HTTP-429 löst den Anbieterwechsel aus.
  Andere Parser-/Netzwerkfehler verwenden die bestehende Fallback-Kette.
  Ein einzelner Provider-Request nutzt maximal drei Sekunden Timeout.
- `/health`, `/api/neral/chat/three-tier/status` und
  `/api/web/search/status` zeigen lokale Diagnoseklassen an, ohne
  externe Abfragen beim Abruf der Status-Endpunkte auszulösen.
- Die Oberfläche akzeptiert den vom Server ausdrücklich bestätigten
  rohen V4.3-9B-Modus, verwechselt ihn aber nicht mit dem LoRA-Adapter.
  Die drei Stufen rendern gelesene Quellen aus `web_sources`.

## Reproduzierbarer Offline-Test

```bash
python -m pytest -q tests/test_neral_v43_live_web_cpu.py \
  tests/test_neral_v43_routes_cpu.py \
  tests/test_neral_v43_bounded_cpu.py \
  tests/test_neral_web_search.py \
  -k 'not test_main_app_inherits_cookie_and_host_origin_guards'
node --check static/neral-preview.js
node tests/test_neral_v21_ui.cjs
```

Die ausgelassene alte Host-/Origin-Integration importiert `app.py`
und benötigt das separate, schwere `torch`-Paket. Sie wird
nicht als bestanden gemeldet. Alle neuen Web-Fälle sind mit
HTTP-Mocks getestet.

**Grenze:** Kein echter Suchanbieter, Windows-SSL-Store, Proxy
oder Self-Hosted-Runner wurde live geprüft. Da die Anbieter
inoffiziell sind, kann ein künftiger HTTP-Block trotz Fallback
nicht durch reine CPU-Tests ausgeschlossen werden.
