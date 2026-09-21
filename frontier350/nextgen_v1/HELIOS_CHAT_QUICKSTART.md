# Helios: echter lokaler Mehr-Nachrichten-Chat (experimentell)

Der neue Einstieg `helios_chat_app.py` lädt **ausschließlich eigene, SHA-geprüfte 350M-Gewichte** in CPU-Inferenz und stellt eine Browseroberfläche unter `http://127.0.0.1:9683/` bereit. Fragen, Modellantworten und ältere Frage-Antwort-Paare werden als Modellkontext übertragen; der Server speichert keine Gesprächsverläufe. Dies ist **keine neue Modellversion** und kein belegter Qualitätsgewinn. Die Qualität des 2B-historischen Checkpoints ist noch unzureichend für einen verlässlichen Chatbot.

## Start auf deinem Windows-PC

Zuerst die Änderungen des Entwicklungszweigs `experiment/phoenix-blind-ab-audit-20260919` in einer separaten Arbeitskopie prüfen, ohne ungesicherte lokale Änderungen zurückzusetzen. Benötigt werden die vorhandenen Python-/Torch-/Tokenizer-Abhängigkeiten aus dem Projekt und die **tatsächlichen eigenen** Checkpoints und Assets auf deinem Rechner. Die Dateien befinden sich bewusst nicht auf GitHub.

Der ältere, als SAFE bezeichnete Checkpoint muss sich unter dem Assets-Root befinden:

```powershell
python -B frontier350/nextgen_v1/helios_chat_app.py --execute --trusted-local-checkpoint --model safe --assets-root "D:/YOUR_REAL_NERAL_ASSET_ROOT" --checkpoint "D:/YOUR_REAL_NERAL_ASSET_ROOT/runs/neral350_autorecovery_neral-autorecovery-v1-20260912-r1_selected/resume.pt"
```

Für den unveränderten historischen 2B-Token-Checkpoint (genauer Original-SHA `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`), wenn er lokal vorhanden und **von dir ausdrücklich als eigenes Checkpoint vertraut** eingestuft ist:

```powershell
python -B frontier350/nextgen_v1/helios_chat_app.py --execute --trusted-local-checkpoint --model continuity-2b --assets-root "D:/YOUR_REAL_NERAL_ASSET_ROOT" --checkpoint "D:/YOUR_PINNED_ORIGINAL/resume.pt"
```

Ersetze diese Beispielpfade durch vorhandene Pfade. Nichts wird heruntergeladen oder als fehlendes Modell simuliert. Öffne anschließend `http://127.0.0.1:9683/` auf **demselben PC**. `Ctrl+C` beendet den Server. Ein anderer Rechner oder das Smartphone kann sich nicht über das Netzwerk verbinden, weil der Server absichtlich nur an `127.0.0.1` gebunden ist.

## Was tatsächlich passiert

- Der Server benutzt den bereits existierenden, SHA-geprüften Modell-Ladepfad. Die Antworten kommen aus den eigenen Modellgewichten, nicht von einer fremden API. Es gibt keine Trainings- oder Checkpoint-Schreibfunktion.
- Der Browser zeigt den Gesprächsverlauf an und schickt ihn zusammen mit der nächsten Nutzerfrage. Das Modell erhält die real kodierte Rollenfolge `User: ... / Assistant: ...`; bei Überschreiten des **512-Token-Kontextfensters einschließlich reservierter Antwort-Tokens** werden nur ganze älteste Frage-Antwort-Paare verworfen und die Anzahl angezeigt. Eine allein zu lange Frage wird abgewiesen.
- Die Antworten sind auf 64 neue Tokens beschränkt, inklusive möglichem EOS. Eine Wiederholungsbremse und der Abbruch vor einer neuen Sprecherrolle sind nur Decoder-Heuristiken; sie machen die Modellqualität nicht automatisch besser. Frühere Nachrichten werden nur innerhalb dieses Tab-Arbeitsspeichers gehalten und beim Reload oder „Neuer Chat“ entfernt. Antworten werden **nicht** auf Server oder GitHub gespeichert.
- HTTP-Protokoll: `GET /health`, `POST /chat` mit JSON `{"messages":[{"role":"user","content":"Hallo"}],"max_new_tokens":64}`. Lokale Host-/Origin-Prüfung, maximale Requestgröße und keine fremden Netzwerkverbindungen durch diese Anwendung.

## Nachweis und Trainingsgrenze

Der isolierte neue Kontext-/Decoderkern hat **9/9 synthetische CPU-Tests** bestanden; die drei Python-Dateien wurden lokal auf Syntax geprüft. Das bedeutet nicht, dass dieses UI mit dem vollständigen privaten Repo, den echten 350M-Gewichten und deiner Hardware bereits erfolgreich gestartet wurde. Für die tatsächliche Modellqualität braucht es echte freie Antworten und einen unabhängig eingefrorenen, zweisprachigen Test.

**Training ist nicht automatisch aktiviert.** Die vorhandene Datei `phoenix_weight_only_pilot.py` kann nach gesonderter, wahrheitsgemäßer Daten-/Holdout-Prüfung und einer SHA-gebundenen Freigabe höchstens 262.144 **neue** Tokens mit **neuem AdamW** trainieren. Exaktes Wiederaufnehmen des alten Optimizers ist aufgrund fehlender Momente ausgeschlossen. Die erforderliche Prüfung und Freigabe liegen noch nicht vor. Keine GPU ist in der ChatGPT-Ausführungsumgebung verfügbar; ein echter 350M-Lauf muss explizit auf dem Rechner mit den privaten Assets erfolgen. Siehe `WEIGHT_ONLY_PILOT_OPERATOR_CONTRACT.md` und Issue #119. Keine synthetischen Test-PASSes als reale Trainingslogs ausgeben.
