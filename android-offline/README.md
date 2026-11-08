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
- Chats bleiben während dieser Vorschau im Arbeitsspeicher der App; noch kein lokaler Chat-Export, kein Kontosystem, kein Vision-/Sprachmodus.
- Das Modell liegt ausschließlich im privaten App-Dateiverzeichnis. Kein Verbindungsversuch zu einem Inferenzserver, keine bezahlte Cloud; die Android-INTERNET-Berechtigung wird ausschließlich beim **expliziten** GGUF-Download verwendet.
- Die App prüft vor dem Laden Dateigröße und GGUF-Header. **SHA256-Pinning und vollständiger Android-Gerätetest sind noch offen**; keine sicherheitstechnische Produktfreigabe.
- CPU-Inferenz kann auf älteren Handys langsam sein oder bei knappen RAM scheitern. Akkuverbrauch ist nicht gemessen.
- Downloads sind nicht resumierbar; Abbruch/Verbindungsfehler entfernt die unvollständige Datei. Bereits fertig heruntergeladene Modelle bleiben erhalten.

## Entwickeln / testen

Benötigt JDK 21, Android SDK 35, NDK 27.2.12479018, Gradle 8.10.2 sowie passende Git/CMake-Tools. `android-offline/libs/llama.kt` wird in CI aus einem **festen** Upstream-Commit geholt, die native llama.cpp-Schicht via `scripts/bootstrap.sh` erzeugt. Der native Build ist bewusst **nicht** mit einem unkontrolliert aktualisierten `main` verknüpft.

```bash
gradle --no-daemon -p android-offline :app:testDebugUnitTest :app:assembleDebug
```

Die App führt keine Modellgewichte im Git-Repository. Android-Abnahme: reales Gerät, Download, Flugmodus, erster Prompt, zweiter Prompt, Abbruch, App-Neustart, Speicher/Hitze.
