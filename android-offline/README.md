# NERAL Offline – kostenlos direkt auf Android

**Ziel:** Eine native Android-Chat-App, die ohne PC, ohne Server und nach einem einmaligen Modelldownload ohne Internet arbeitet. Kein Abonnement und keine API-Kosten.

## Android APK laden

1. Im GitHub-Repositorium unter **Actions → NERAL Android Offline APK** den neuesten *grünen* Lauf öffnen.
2. Unten unter **Artifacts** das Paket `NERAL-Android-Offline-Preview-APK` laden. Das ZIP entpacken und `app-debug.apk` auf dem eigenen Android-Gerät installieren.
3. In der App auf **Modell laden** tippen (einmal ca. 639 MB Internet-Download, Datenverbrauch je nach Mobilfunktarif möglich).
4. Anschließend ohne WLAN/Mobilfunk und **mit ausgeschaltetem Windows-PC** chatten. **Die Antwortgenerierung ist vollständig lokal.**

**Hinweis:** Bis der Workflow grün und ein echter Android-Gerätetest bestanden ist, ist die APK *ein unbestätigter Entwicklungskandidat*, keine fertige Release-Version. Die Installation von APKs außerhalb des Play Store erfordert ggf. eine einmalige Android-Freigabe für die installierende App.

## Grenzen und Herkunft

- Läuft als native Android-App auf **arm64 / Android 10+** (minSdk 29).
- Native CPU-Inferenz durch [llama.kt](https://github.com/tensai-app/llama.kt) / llama.cpp, auf Commit `47dde2d0f2b423707cb43e43b62d2aee09309680` gepinnt.
- **Modell:** Unverändertes [Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), Q8_0, Apache-2.0. Bei der Installation nur die App; nach ausdrücklichem Tippen wird die offizielle GGUF-Datei via HTTPS geladen.
- Dies ist **nicht** das trainierte NERAL-LoRA-Modell vom PC und keine Überlegenheit über Qwen. Es ist deutlich kleiner und bei schwierigen Aufgaben schwächer.
- Der Chatverlauf wird begrenzt im privaten App-Speicher gesichert und beim Neustart wiederhergestellt. Antworten lassen sich kopieren; „Neuer Chat“ löscht den gespeicherten Verlauf. Noch kein Export, Kontosystem, Vision- oder Sprachmodus.
- Das Modell liegt ausschließlich im privaten App-Dateiverzeichnis. Kein Verbindungsversuch zu einem Inferenzserver, keine bezahlte Cloud; die Android-INTERNET-Berechtigung wird ausschließlich beim **expliziten** GGUF-Download verwendet.
- Die App prüft Dateigröße, GGUF-Header und die vom Modellhersteller veröffentlichte SHA-256-Prüfsumme `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` vor dem Laden. Die vollständige Android-Geräteabnahme ist weiter offen; keine sicherheitstechnische Produktfreigabe.
- CPU-Inferenz kann auf älteren Handys langsam sein oder bei knappen RAM scheitern. Akkuverbrauch ist nicht gemessen.
- Downloads sind noch nicht fortsetzbar; bei Verbindungsfehlern muss der Download wiederholt werden. Bereits fertig heruntergeladene und verifizierte Modelle bleiben erhalten. Für eine neue APK aus GitHub Actions muss möglicherweise die alte Debug-APK wegen unterschiedlicher Debug-Signaturen deinstalliert werden; das löscht auch deren lokale Chats und Modell und ist eine Preview-Einschränkung.

## Status der tatsächlichen Funktionalität

Diese Android-Preview bietet einen lokalen Chat mit Antworten, Speicherung, Kopieren und Abbruch. Sie hat **keine** vollständige Funktionsparität mit der Windows-NERAL-App (keine Websuche, keine Datei-/Bildanalyse, keine Plugins, kein Sprachchat, kein trainierter NERAL-LoRA). Ein bestandenes APK-Build beweist noch nicht, dass das Modell auf jedem Android-Handy korrekt antwortet. Test auf echtem Gerät mit mindestens zwei aufeinanderfolgenden Antworten und Flugmodus ist weiterhin erforderlich.

## Entwickeln / testen

Benötigt JDK 21, Android SDK 35, NDK 27.2.12479018, Gradle 8.10.2 sowie passende Git/CMake-Tools. `android-offline/libs/llama.kt` wird in CI aus einem **festen** Upstream-Commit geholt, die native llama.cpp-Schicht via `scripts/bootstrap.sh` erzeugt. Der native Build ist bewusst **nicht** mit einem unkontrolliert aktualisierten `main` verknüpft.

```bash
gradle --no-daemon -p android-offline :app:testDebugUnitTest :app:assembleDebug
```

Die App führt keine Modellgewichte im Git-Repository. Android-Abnahme: reales Gerät, Download, Flugmodus, erster Prompt, zweiter Prompt, Abbruch, App-Neustart, Speicher/Hitze.

## Android-Startabsturz – Recovery v0.3

In v0.3 startet zuerst eine native, bewusst ohne Compose und ohne Modellbibliothek realisierte Recovery-Oberfläche. **NERAL Chat öffnen** startet erst dann die Compose-Chatanwendung. Ein von Android/Kotlin ausgelöster unbehandelter Fehler wird im privaten App-Speicher abgelegt und bei einem erneuten Start als kopierbare **Fehlerdiagnose** angeboten; nichts wird automatisch hochgeladen. JNI-`UnsatisfiedLinkError` beim Modellstart wird im Chat abgefangen.

Die neue Testversion nutzt vorübergehend `applicationId = ai.neral.offline.recovery` und das sichtbare Label **NERAL Offline 0.3**, weil GitHub-Actions-Debug-APKs früherer Läufe nicht zwingend dieselbe Android-Signatur haben. Sie kann daher neben der alten defekten Preview installiert werden. Das bedeutet einen eigenen privaten Speicher und einmaliges erneutes Herunterladen der Modellgewichte; keine automatisierte Migration. Nicht als signierte Release-/Store-Version verwenden.

Das Starten der Recovery-Oberfläche im Android-Emulator und auf dem konkret betroffenen Handy ist noch nicht bestätigt. **Build-Erfolg ist keine reale Crash-Reproduktion.** Wenn „NERAL Chat öffnen“ abstürzt, die neue App erneut öffnen und über „Fehlerdiagnose kopieren“ den Stacktrace bereitstellen.
