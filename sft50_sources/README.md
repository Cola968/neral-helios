# Neral-50M Stage D source contract

Stage D instruction data stays local. The repository contains the builder and schema, not the raw corpus.

Create `sft50_sources/manifest.json` and one or more local JSONL files. Every source entry must declare:

- `id`
- `file`
- exact `sha256`
- `license`
- `license_url`
- `provenance_url`
- `language`: `de` or `en`
- `domain`: one of `chat`, `explanation`, `code`, `math`, `science`, `focus`, `translation`, `reasoning`, `summary`, `safety`, `format`, `knowledge`
- `target_fraction`

All `target_fraction` values must sum to `1.0`.

Example manifest:

```json
{
  "version": 1,
  "sources": [
    {
      "id": "example-human-dialogue-en",
      "file": "dialogue_en.jsonl",
      "sha256": "REPLACE_WITH_EXACT_64_HEX_SHA256",
      "license": "REPLACE_WITH_VERIFIED_LICENSE",
      "license_url": "https://example.invalid/license",
      "provenance_url": "https://example.invalid/source",
      "language": "en",
      "domain": "chat",
      "target_fraction": 1.0
    }
  ]
}
```

Each JSONL row must have a stable `id` plus either:

```json
{"id":"row-1","prompt":"Explain orbital velocity simply.","answer":"Orbital velocity is ..."}
```

or exactly one user/assistant pair:

```json
{"id":"row-2","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Quality rules

`prepare_sft50.py` rejects malformed Unicode/control characters, likely credentials/contact data, extreme repeated-character runs, overlong examples, exact pair duplicates and exact normalized long literals from the frozen Neral benchmark suites.

The builder performs the split before selecting the train quota, so validation/test/holdout examples are never pulled into the training quota.

Use sources only after verifying their actual license and redistribution/training terms. Do not put private source data or generated `sft50_data/` into Git.
