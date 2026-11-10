# PR #505: Turn-0 preparation

The old grammar required only `kind` and explicitly allowed `none`. Thus
`{"kind":"none"}` could finish in about six tokens (`done_reason=stop`),
despite the 384-token budget. The parser returned `None` and TIR selected
`extraction_fallback` without calling the solver.

The extractor now sends a closed JSON Schema through Ollama's `format`:
generic English/German Boolean/order task cues plus a count/feasibility query
remove the `none` branch. A formal candidate requires `complete=true`,
nonempty variable declarations (or at least two ordering items), at least one
constraint, and source evidence. Variables use a list of `{name,domain}`
objects because llama.cpp's grammar converter does not enforce dynamic-object
`minProperties`/`propertyNames`. The parser normalizes the list to the existing
solver map and rejects duplicate declarations. Boolean expression depth,
operators, arity and domains are bounded. Exact quotations, source variable
checks and solver limits remain mandatory; successful solving does not verify
the natural-language translation. Unsupported or truncated proposals can still
fall back. Raw-Qwen sampling remains unchanged.

The prompt contains exactly one executable Boolean example (2 assignments)
and one ordering example (3 orders). CPU tests parse the actual prompt examples,
validate the outgoing format, and execute them through the real parser/solver.

## Tomorrow: optional local one-task smoke

Run only after the GPU PC is on, with no concurrent GPU workflow. In the
checkout of the final PR commit, use the same Python environment as TIR12:

```powershell
$env:NERAL_EXPERIMENTAL_TIR = '1'
python -B -m scripts.neral_tir_logic_smoke --authorize NERAL_TIR_LOGIC_SMOKE_GPU_MANUAL_ONLY --target-sha (git rev-parse HEAD)
```

This runs one Boolean extraction and the real solver; it makes no raw-Qwen
baseline or fallback inference. The extraction HTTP timeout is 30 seconds;
cold model loading, inventory and unloading can add time. It checks the exact
checkout SHA, clean tracked source and original model digest. Exit 0 requires
`successful_tool_calls >= 1` and the correct answer. Output contains scalars
only. Exit 1 means extraction/solver smoke failed; exit 2 means setup failed.
No smoke execution or GPU workflow is scheduled by this change. Start the full
manual TIR12 gate separately with the final PR SHA after a successful smoke.

The pre-existing `neral-agent-live-pc.yml` PR trigger used to start an
unguarded self-hosted GPU job. PRs now run those contracts on hosted Linux
and Windows CPUs. The live PC job requires a separate manual dispatch on
`main` with `NERAL_AGENT_PC_GPU_MANUAL_ONLY`; a PR push cannot queue it.

CPU verification:

```sh
python -m pip install -r requirements-neral-preview.txt -r requirements-tests-turn0.txt
python -B -m unittest discover -s tests -p 'test_neral_tir_turn0.py' -v
python -B -m unittest discover -s tests -p 'test_neral_tir_logic_smoke.py' -v
```
