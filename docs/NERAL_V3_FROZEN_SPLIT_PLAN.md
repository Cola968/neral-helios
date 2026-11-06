# NERAL: reproduzierbare Rank-16/Rank-32-Vergleichsbasis

**Eingangsdaten:** Ausschließlich der unveränderte, am 6. November 2026
auf dem Windows-Runner nachgeprüfte 384er-Core-/Algebra-v3-Kandidat.

- SHA-256 Datensatz: `4b0e54712ea889cadd98eb668d315f3eb8647a7b9ac2209a1ed9942f7bbd3941`
- SHA-256 Review-Manifest: `db1a460da4a19057877a32aa4bccc5814dd819442c4b678e7437ca9dca30771d`
- Ursprung: <https://github.com/Cola968/neral-helios/actions/runs/42280829676>.

**Stratifizierte Aufteilung:** Training 320 Zeilen, Validierung 64 Zeilen,
reproduzierbare Auswahl über SHA-256 + unveränderlichen Versions-Salt;
keine doppelten Zeilenindizes. Alle 384 Originalzeilen behalten ihre
rechnerische oder ursprüngliche Quellenprüfung. Die neue per-split
Review-Datei enthält für jede Zeile den ursprünglichen Quellindex
`source_line` und die belegte Quellen-Prüfsumme. Der unveränderte
QLoRA-Prüfer validiert beide Dateien auf dem Windows-Runner.

| Kategorie | Training | Validierung |
|---|---:|---:|
| NERAL-Core | 72 | 14 |
| Historische Mathematik | 34 | 6 |
| Frühere Gleichungen | 58 | 12 |
| Neue verifizierte Gleichungen | 84 | 16 |
| Andere Mathematik | 72 | 16 |
| **Summe** | **320** | **64** |

**Grenzen:** Durch die ähnlichen generierten Rechenvorlagen ist die
Validierung *kein unabhängiger Generalisierungstest*; sie dient der
Überanpassungskontrolle beim späteren Rangvergleich. Die bereits
öffentlich diskutierten 80 Mathematik- und 36 Zusatzaufgaben sind
Entwicklungschecks und ebenfalls nicht blind. Für die Release-Freigabe
werden gesonderte, untrainierte Coding-/Tool-/Core-Aufgaben benötigt.

**Nächste Schritte, ausdrücklich nicht automatisch aktiviert:**
Ein Rank-16- und Rank-32-QLoRA-Versuch müssen dieselben 320 Trainings-
und 64 Validierungszeilen, dieselbe effektive Batchgröße, denselben
Seed, Schrittzahl und Lernrate nutzen. Rank 32 benötigt möglicherweise
mehr VRAM als für 120 Steps auf einer RTX 4070 verfügbar ist; die
historisch verifizierten 80.740.352 trainierbaren Parameter
sind nur ein Kapazitätsnachweis, keine Quality-Prognose.
Die bekannte Begrenzung von Rank 32 auf drei Smoke-Schritte darf
nicht stillschweigend ausgeschaltet werden.

Workflow `.github/workflows/neral-v3-frozen-split.yml` ist nur ein
offline-lokaler Datenprüfer. Kein GPU-Training, keine Modell-Aktivierung
und keine Dateiübertragung des eigentlichen Trainingskorpus.
