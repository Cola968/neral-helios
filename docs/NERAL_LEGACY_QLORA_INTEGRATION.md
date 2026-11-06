# NERAL: NeralTalk350 → Qwen2.5-Coder-7B QLoRA

Status: technisch integriert als lokaler Trainingskandidat. Noch nicht fachlich freigegeben oder trainiert.

Ausgangspunkt: 430 lokale NeralTalk350-Beispiele aus GitHub Run #42242745430.
Erwarteter Kandidat-SHA256: 50c3c200913f6e40ed4b3975de39ecba795c1839b9983fdcd3029095b716fa9a.

Der manuelle Workflow .github/workflows/neral-legacy-qlora-integration.yml überprüft offline:
- Kandidat- und Original-Train/Val/Test-Digests; train-only-Auswahl wird exakt rekonstruiert.
- QLoRA-Datenvertrag des gepinnten Trainers a082738f9145722dec443141e5d7801a3ead93af.
- Überlappungen mit den 8 fixierten QLoRA-Holdouts und 86 alten Trainingsbeispielen.
- Echten Qwen-Tokenizer, Chat-Template mit Abschlussgrenze und 512-Token-Maximum.

Nur der zahlen- und hashbasierte summary.json-Bericht wird hochgeladen.
Der bereinigte Datensatz train_candidate.jsonl und die review_queue.jsonl verbleiben lokal unter
NERAL-QLoRA-Research/legacy-integration-<run_id>-<attempt>/.

WICHTIG: approved_for_training=false. Die bisherige verify_review-Sperre darf nicht umgangen werden.
Vor jedem Training sind zeilenweise inhaltliche Quellenprüfung, Near-Duplicate-/Benchmark-Audit,
ein zulässiges Review-Manifest und ein neu autorisierter GPU-Test mit frischem Adapter-Ausgabeverzeichnis erforderlich.
Der aktuelle Basis- und LoRA-Adapter und Arctus350 werden nicht verändert.

CPU-Tests: python -m unittest discover -s tests -p test_neral_legacy_qlora_gate.py -v
