# NERAL v3 – Auswertung ohne neues Training

Neuer GitHub-Workflow: **NERAL v3 saved adapter evaluation ONLY (no training)**.

Start auf GitHub unter Actions → Workflow auswählen → Run workflow auf `main` → `authorize` = `EVALUATE_NERAL_V3_SAVED_80`.

Der Windows-Self-hosted-Runner `neral-gpu` muss online sein. Der Workflow nutzt nur die lokal gespeicherten Kandidaten aus dem geprüften Lauf **42297190703**, Versuch 1, und die Audits aus **42296646197**, Versuch 1. Keine LoRA-Trainingsschritte oder Downloads.

Für Qwen-Basis, NERAL Rank16 und Rank32 startet **je ein neuer Python-Prozess**. Vor dem Laden: gc.collect, torch.cuda.empty_cache, BF16-Prüfung, mindestens 9000 MiB VRAM und höchstens 20 % GPU-Last; PyTorch-Allocator auf 85 % des Gesamtspeichers beschränkt. Nach dem Lauf wird das Modell gelöscht und der CUDA-Cache geleert; die Prozessbeendigung erzwingt vollständige Freigabe von CUDA-Kontexten. Das ist robuster als eine CUDA-Bereinigung innerhalb eines einzigen Langzeitprozesses, aber keine absolute OOM-Garantie.

Verglichen werden auf identischen Fragen **64 Validierungsbeispiele**, **80 Mathematikfragen**, **36 weitere bereits verwendete Entwicklungsfragen** sowie **20 zusätzliche interne Entwicklungsfragen**. Die Ergebnisse sind nicht blind oder unabhängig. Die Modellgewichte und Rohantworten bleiben auf dem PC; ausschließlich Zähler und SHA-256-Prüfsummen werden in GitHub Actions hochgeladen.

Geprüfte Adapter-SHA256:
- Rank16: `0f0b6b48e615d59f2dff0754dbaab902604beb66814871888a22d3210c51202b`
- Rank32: `cb67b0fb3c9b3eb84bb8353e7ff00cf373c49a921a23ac94848d3a461047b97d`

Für den Vergleich mit 600 Generierungen ist eine erheblich längere Laufzeit als der Vier-Minuten-Trainingsabschnitt möglich. Bei unzureichendem GPU-Speicher wird sicher abgebrochen; der Trainingsworkflow darf nicht wiederholt werden. Keine automatische Release-Freigabe.
