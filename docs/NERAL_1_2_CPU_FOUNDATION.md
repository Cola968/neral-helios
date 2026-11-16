# NERAL 1.2 – CPU preparation (isolated)

This package is opt-in, **not** connected to the production model, router or web API.
The source commit is 7fb05452fa7588191f9a127b11511a2ec312baf1.
No GPU, Ollama, Docker or network access is required to run its tests.

```bash
python -m pip install pytest==8.4.2 sympy==1.14.0 networkx==3.6.1 z3-solver==4.15.3.0
python -m pytest -q tests/cpu_foundation
python -B scripts/neral_cpu_foundation_build.py --output cpu-artifacts
```

Evidence categories: Z3 proves satisfiability/UNSAT or absence of a better integer optimum, SymPy checks symbolic identities and candidate substitutions, and NetworkX checks graph witnesses with independent algorithms where possible. A successful Python subprocess or 1-step AST repair is **not** independent proof and is rejected by the training exporter. No private thought traces or existing blind benchmark records are ingested.

The exported `helios_terra_v2.jsonl` and `luna_distill_v1.jsonl` are train-only aliases. Validation/test splits are separate. The examples are synthetic, template-derived, and **not independent blind evaluation data**. Do not train on their validation/test files. Exported 'proof' means the stated, bounded formalized task was checked, not that an LLM correctly translated arbitrary natural language.

Limitations: SymPy uses a spawned worker process with a deadline but is not a system sandbox for adversarial multi-tenant workloads; graph Hamilton search is exhaustive only up to 10 vertices; web endpoints are opt-in and mock-tested; the DuckDuckGo zero-click JSON endpoint is not equivalent to full search; local Memory requires explicit consent; the vector-RAG hook requires a separately chosen embedding provider and is unpopulated by default.

Integration after the 1.0/1.1 merge needs a separate PR, schema audits and Windows/RTX verification.
