# NERAL v2: genehmigter 36er-Entwicklungstest (read-only)

**Freigabe:** Nur Ausführung des unveränderten, nicht verblindeten Entwicklungsbenchmarks.
**Keine Modellfreigabe, kein weiteres Training.**

- Testquelle: `scripts/neral_v2_generalization_probe.py`
- Kanonischer SHA-256 des 36-Fälle-Katalogs **einschließlich SYSTEM-Anweisung**:
  `971f17ed80991a7a6791cb525a4e0ba95f5cd60ec066816fbca0f2ca190c18c1`.
  Kanonisierung: UTF-8 JSON mit `schema=1`, `cases`, `system`,
  `sort_keys=True`, `ensure_ascii=False`, `separators=(',', ':')`.
- Qwen-Basis-Revisionspin: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- Bestehender NERAL-R32-Adapter: `090aac244e1057c4a0a3a51400ecf0ca3f4619d0ecb0b871f5da27c3825c6a39`
- Neuer Mathematik-v2-R16-Adapter: `9c89979280b09e3818f4f408dcce909c1563be7b0c24c949825ec4802600e758`
- Originaler Trainingslauf: `42264245416`; kein erneutes Training.
- Ausschließlich Autorisierung: `RUN_READONLY_V2_GENERALIZATION`.
- 24 Mathematik- und 12 einfache exakte Text-Anweisungsfälle; **keine** Aussage über Coding, Sicherheit oder allgemeinen Agentenmodus.
- Bestehende Pflichtprüfungen (Windows-Runner, Modelldatei-SHA, Adapter-SHA, CUDA/VRAM, Datenschutzbericht, Ausgabeverzeichnis) bleiben aktiv.
- Die vorherige angebliche Blockade geschah **vor** GitHub Actions: Bei Auditbeginn existierten keinerlei Runs dieses Workflows. Deshalb ist weder ein Hash-, Pfad- noch ein Berechtigungsfehler im Runner als Ursache belegt. Es wäre unzulässig, hierzu einen präzisen GitHub-Fehler zu erfinden.
- Die bisherigen 80 Mathematik-Aufgaben sind Entwicklungstests, kein unabhängiger Release-Holdout.

Durchführung: Workflow `.github/workflows/neral-v2-generalization.yml` mit `v2_run_id=42264245416` und korrekter Autorisierung starten. Nur Zähler/Hashes hochladen.
