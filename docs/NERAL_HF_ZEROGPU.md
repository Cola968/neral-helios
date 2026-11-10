# NERAL: ohne Windows-PC chatten – ZeroGPU-Vorbereitung

**Status: Vorbereitung, nicht bereitgestellt. Es gibt noch keinen funktionierenden Live-Link.**

## Warum GitHub allein keinen Chat ermöglicht

GitHub enthält kein freigegebenes NERAL-7B-/LoRA-Gewichtsartefakt. Die Rank-16- und Rank-32-Adapter wurden auf dem Windows-PC trainiert und dort gespeichert. Die Actions-Berichte enthalten ausschließlich Zahlen und Hashes. GitHub Pages ist statisch und GitHub Actions ist kein dauerhafter GPU-Inferenzdienst.

## Möglicher Weg: Hugging Face ZeroGPU

ZeroGPU stellt Gradio-Apps eine geteilte GPU pro Anfrage zur Verfügung. Berechtigte kostenlose HF-Personenkonten (verifizierte E-Mail, älter als 30 Tage, guter Status) dürfen bis zu zwei ZeroGPU-Spaces hosten. Kontingente, Warteschlange und Modellkompatibilität können den Test begrenzen. Quelle: https://huggingface.co/docs/hub/spaces-zerogpu

1. **Wenn dein PC später wieder verfügbar ist:** Den geprüften NERAL-LoRA-Adapter (adapter_model.safetensors und adapter_config.json) in ein *privates* Hugging-Face-Modellrepository übertragen. Den v3-Kandidaten nicht als final freigeben, bevor dessen unabhängige Evaluation besteht. Keine Trainingsprompts, privaten Dateien oder Geheimnisse hochladen.
2. Ein **privates** Hugging-Face-Space mit SDK Gradio und Hardware ZeroGPU anlegen.
3. Den Inhalt von cloud/zerogpu (app.py, policy.py, requirements.txt, README.md) direkt ins Hauptverzeichnis dieses Space-Repos kopieren; die README-Datei enthält erforderliche Space-Metadaten.
4. NERAL_ADAPTER_REPO als Space-Variable auf das private Adapterrepo (owner/name) setzen. Einen feingranularen, *nur lesenden* Hugging-Face-Zugriffstoken für dieses Repo als Secret HF_TOKEN eintragen. Keine Tokens in GitHub oder Chat posten.
5. Nur dann testen, wenn eine echte Adapter-Inferenz startet. Bei fehlendem Adapter wird die Demo bewusst nicht geladen.

## Grenze und Datenschutz

- Das aktuell verbundene HF-Konto kann über die ChatGPT-Verbindung gelesen, aber nicht verändert werden. Ein Space wurde deshalb hier weder erstellt noch veröffentlicht.
- Cloud-Eingaben werden beim Betreiber Hugging Face verarbeitet; das ist NICHT gleichbedeutend mit lokalem Datenschutz. Keine externe Qwen-Inferenz-API wird verwendet.
- Der Code hat keine Account-Isolation, keine Cloud-Abrechnung, keine Agenten und keine Computersteuerung. Deshalb **private Testumgebung**, keine öffentliche Beta.
- Das CUDA-Quantisierungssystem bitsandbytes + PEFT auf ZeroGPU wurde nicht live getestet; es kann beim Deployment scheitern.
- Foundation Qwen2.5-Coder-7B-Instruct; vor Nutzung Adapter-/Foundation-Kompatibilität mit dem lokal verifizierten Original abgleichen und bei Bedarf eine passende feste Revision pinnen.
- Es wurden keine kostenpflichtigen GPU-Dienste aktiviert und keine LoRA-Gewichte veröffentlicht.

## Prüfungen

Die reinen CPU-Vertragsprüfungen (nicht der GPU-Inferenztest):
python -m unittest discover -s tests -p 'test_zerogpu_policy.py' -v
