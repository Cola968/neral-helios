# NERAL Assistant: sorgfältiges Denken (Qwen3.5, optional)

## Ausgangspunkt

NERAL Assistant nutzt Qwen3.5-9B/4B als lokales **externes** Open-Weight-Modell
via Ollama. Arctus350 ist weiterhin das separate eigene Forschungsmodell.
Issue #195 / PR #203 sind die Integrationsbasis. Die Änderungen in diesem
Draft trainieren **keine** eigenen Gewichte.

Der bisherige Assistent lief mit `think=false`, 4096 UTF-8-Bytes
Gesprächskontext und üblicherweise 256 Modell-Ausgabetoken. Das kann längere
Aufgaben und zusammenhängende Konversationen unnötig beschneiden.

| Einstellung | Standard | „Sorgfältig denken“ |
|---|---|---|
| Ollama `think` | explizite Serverkonfiguration, normalerweise `false` | `true` |
| Antwortbudget | 128/256/512 (UI) | 1024 Token |
| Ollama `num_ctx` | 4096 Modelltoken | 8192 Modelltoken |
| Max. Verlauf + Anfrage | 4096 UTF-8-Bytes | 8192 UTF-8-Bytes |
| Pro-Nachricht-Limit | 4096 UTF-8-Bytes | 4096 UTF-8-Bytes |
| Backend | derselbe Qwen-Provider | derselbe Qwen-Provider |
| Kosten | lokale Inferenz | lokale Inferenz, mehr Rechenzeit möglich |

**Warum diese beiden Einheiten?** Die Anwendung begrenzt Eingabetext in
UTF-8-Bytes als konservativen Schutz, während Ollama den Modellkontext in
Token verwaltet. Byte- und Modelltokenwerte sind nicht gleichzusetzen.

Der Modus wird im Qwen-Chat ausdrücklich über **Sorgfältig denken**
eingeschaltet. Er ist nicht stillschweigend aktiv und erscheint bei
Arctus350 nicht. Die API erhält `quality=deep`; nur in diesem Zustand
darf das Antwortbudget auf 1024 steigen. Der bestehende native Modellpfad
bleibt bei maximal 512 Token. Anfragen mit fremden Qualitätsangaben
werden vor der Inferenz abgelehnt.

### Technische Sicherheit und Grenzen

- Kein Auto-Download, keine Cloud, keine neuen Plugins, keine Shell-Tools.
  Einziger Ollama-Endpunkt bleibt der feste Loopback-Host.
- Qwen-Kennzeichnung `source=external-model` und `base_model=qwen3.5:9b|4b`
  bleibt erhalten. Der neue Modus ist **keine native Arctus-Verbesserung**.
- Es gibt weiterhin **kein Streaming und keinen echten Stop-Knopf** für
  diesen Ollama-Pfad. Der lokale HTTP-Aufruf hat ein Timeout.
- Höheres Antwortbudget und längerer Kontext können VRAM/Arbeitsspeicher,
  Laufzeit und Stromverbrauch erhöhen. Gerade RTX 4070 / 12 GB:
  **nicht** ohne echten Windows-Test als ausreichend oder schneller bewerten.
- Der Denkmodus kann korrektere Antworten begünstigen, garantiert aber
  keinen IQ-Sprung und verhindert weder Halluzinationen noch Rechenfehler.
  Bei sehr langen Denkfolgen kann das Ausgabelimit vor einer endgültigen
  Antwort erreicht werden – dann ist die Antwort korrekt als Fehler
  behandelt und darf nicht als erfolgreicher Chat gespeichert werden.

### Technische Prüfungen

Offline/CI:
```bash
python -m pytest -q tests/test_issue195_assistant.py tests/test_neral_assistant.py
node --test tests/test_issue195_assistant.js
node tests/test_issue195_assistant_browser.cjs
```

Der Browserdurchlauf nutzt echten lokalen Flask-Code und einen **simulierten
Ollama-HTTP-Endpunkt**. Er ist kein echter Qwen-Qualitätstest.

Vor Freigabe auf dem eigenen PC: dieselben zurückgehaltenen Deutsch-,
Mathe-, Kontext- und Coding-Fragen jeweils in Standard/Deep über die echte
NERAL-Oberfläche an Qwen senden; Antwortgüte, Fehlerrate, tatsächliche
Laufzeit, Grafikspeicher und Stabilität getrennt erfassen. Kein Test wird
ohne Freigabe automatisch auf dem Nutzer-PC ausgelöst.

Offizielle Referenzen:
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://ollama.com/library/qwen3.5
- https://github.com/ollama/ollama/blob/main/docs/api.md
