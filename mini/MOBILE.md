# NERAL Mini auf Android – private Vorschau

NERAL läuft auf deinem Windows-PC. Dein Smartphone ist die verschlüsselte
Chatoberfläche. **Die Qwen-Gewichte und der NERAL-LoRA-Adapter verlassen
den PC nicht.** Nachrichten werden über eine HTTPS-Verbindung im
privaten WLAN übertragen. Keine OpenAI-, Qwen- oder Hugging-Face-API.

## Einrichtung

1. Auf dem **PC** das ZIP **NERAL-Mini-Mobile-Windows** komplett entpacken.
2. **Start NERAL Mini Mobile.cmd** doppelt anklicken. Der Starter prüft
   die vorhandene GPU-Python-Umgebung und öffnet nach der Einrichtung
   die Datei **local_state\mini_mobile\HANDY-ZUGANG.txt** am PC.
3. Handy und PC ins **gleiche private WLAN** bringen. Gast-WLAN mit
   Geräte-Isolierung funktioniert möglicherweise nicht.
4. Die Datei **local_state\mini_mobile\Neral-Android-CA.crt**
   vertrauenswürdig auf dein eigenes Android-Handy übertragen.
   Sie enthält nur ein öffentliches CA-Zertifikat, **nie** server.key
   oder config.json übertragen.
5. Unter Android das Zertifikat als CA installieren. Je nach Gerät:
   Einstellungen → Sicherheit → Weitere Sicherheitseinstellungen →
   Verschlüsselung und Anmeldedaten → Zertifikat installieren →
   CA-Zertifikat. Android zeigt eine Warnung, weil zusätzliche
   Zertifizierungsstellen HTTPS-Verbindungen prüfen können.
   Installiere das Zertifikat nur, wenn es von **deinem PC** stammt.
6. Die HTTPS-Adresse aus HANDY-ZUGANG.txt in Chrome auf dem
   Handy öffnen, z. B. https://192.168.1.42:9444/ – die IP-Adresse
   wird auf deinem eigenen PC ermittelt, dieses Beispiel ist **nicht**
   deine tatsächliche IP. Den Kopplungscode aus derselben Datei
   einmal eingeben und **NERAL Mini öffnen** drücken.
7. Im Chrome-Menü kannst du **Zum Startbildschirm hinzufügen**
   wählen. Das ist zunächst eine Website-Verknüpfung, keine APK
   und keine offline laufende Android-KI.

Der PC und das Starter-Fenster müssen während des Chats eingeschaltet
bleiben. Der Browser auf dem Handy kann dann Nachrichten schicken.
Die lokale Modellinitialisierung bei der ersten Frage kann dauern.

## Probleme beim Verbindungsaufbau

- **HTTPS-Adresse lädt nicht:** PC und Handy ins gleiche WLAN;
  PC-Firewallzugriff auf TCP-Port 9444 **nur im privaten Netzwerk**
  und nur aus dem lokalen Subnetz zulassen. Keine Portfreigabe am Router.
- **Zertifikatsfehler:** Das korrekte eigene CA-Zertifikat übertragen
  und installieren. Niemals Chrome-Sicherheitswarnungen umgehen.
- **Alte IP-Adresse nach WLAN-Wechsel:** Der Zugang ist an die
  IP-Adresse im Zertifikat gebunden. Sichere den bisherigen lokalen
  Zugangsordner, richte danach einen neuen Zugang ein und entferne
  das alte CA-Zertifikat vom Handy.
- **NERAL meldet zu wenig GPU-Speicher:** Beende andere Qwen-Server
  und Trainingsjobs. Die mobile und die lokale App teilen die RTX-GPU.
- **Python-Paket fehlt:** Vorhandene NERAL-QLoRA-GPU-Umgebung prüfen.
  Für die erste Zertifikats-Einrichtung ist zusätzlich das lokal
  installierte Python-Paket cryptography erforderlich.

## Grenzen

- Betrieb ausschließlich auf einer privaten IPv4-Adresse,
  Standard-Port 9444, mit CA-signiertem lokalem HTTPS.
- Ein 128-Bit-Zufallscode koppelt das Handy. Höchstens 5
  Kopplungsversuche pro Minute pro IP-Adresse; Sitzungsdauer
  8 Stunden. Nach einem PC-Neustart ist eine neue Kopplung nötig.
- HTTP-Host, HTTPS-Origin und Sitzungstoken werden überprüft;
  lokale Projektdateien und Trainingsdaten sind über den
  mobilen Endpunkt nicht abrufbar.
- Kein Online-Betrieb außerhalb deines WLANs, keine Routeröffnung,
  keine Websuche und kein Computer Use in dieser Vorschau.
- Die Modellantwort stammt vom experimentellen NERAL-LoRA auf
  Qwen2.5-Coder-7B als Foundation, sofern die lokal verifizierten
  Modellgewichte vorhanden und ladbar sind.
- Keine behauptete allgemeine Intelligenzverbesserung,
  kein Streaming und keine Garantie für eine erfolgreiche
  GPU-Inferenz ohne echten PC-Abnahmetest.

Wird NERAL Mobile nicht mehr genutzt, entferne das eigens installierte
NERAL-CA-Zertifikat wieder aus Android. Das Ablaufdatum der
Server-Zertifikate liegt bei höchstens 180 Tagen.
